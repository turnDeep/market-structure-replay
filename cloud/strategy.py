"""Release identity and the sole v0.12 strategy change; vendor code stays frozen."""
from dataclasses import asdict

STRATEGY_VERSION = "v0.12"
BASE_STAGE = "rev11_without_continuation"


def validate_strategy(metadata):
    rules = metadata["legacy_rules"]
    if (rules["buy_reference_radius"], rules["emergency_buffer_atr"], rules["major_decline_bars"]) != (1, .10, 15):
        raise ValueError("v0.12 requires radius=1, emergency buffer=.10, decline bars=15")
    if metadata["campaign_config"]["enable_continuation_buy"]:
        raise ValueError("Ordinary continuation BUY must remain disabled")


def stamp_metadata(metadata):
    validate_strategy(metadata)
    metadata.update(strategy_version=STRATEGY_VERSION, strategy_base_stage=BASE_STAGE,
                    strategy_change={"legacy_rules.buy_reference_radius": {"from": 2, "to": 1}},
                    extra_supplied_pivot_radii=[1])
    return metadata


def install_pivot_supply():
    """Supply radius-one strict extrema at origin+1, identical to tested alpha.

    Existing radius-two/five confirmations remain untouched. No pivot is supplied
    before its right-hand bar is observed; ties are rejected, as in the study.
    Install once in the market-isolated interpreter before creating the source.
    """
    import numpy as np
    from nq_ict_campaign_examples_20260921.source import CampaignSource
    from nq_ict_retest_trade_state_machine_20260915 import Pivot, number
    if getattr(CampaignSource, "_v012_pivot_supply", False):
        return
    native = CampaignSource.confirm_pivots

    def confirm(self):
        new = native(self)
        if self.rules.buy_reference_radius != 1:
            raise ValueError("Radius-one adapter used with a different strategy")
        origin = self.i - 1
        if origin < 1:
            return new
        window = self.d.iloc[origin-1:self.i+1]
        row = self.d.iloc[origin]
        for kind, column, extremum in [("high", "high", max), ("low", "low", min)]:
            values = window[column].to_numpy(float)
            value = float(row[column])
            if value != extremum(values) or np.count_nonzero(values == value) != 1:
                continue
            pivot = Pivot(f"pivot:1:{kind}:{row.datetime.date()}", kind, 1, origin,
                          row.datetime, self.t, value, number(row.atr_prior))
            self.pivots.append(pivot)
            new.append(pivot)
            self.log("pivot_confirmed", pivot.id, pivot=asdict(pivot))
        return new

    CampaignSource.confirm_pivots = confirm
    CampaignSource._v012_pivot_supply = True
