// Correct only the obsolete stock title through the public shell title API.
// Never clear storage, replace the app ID, or rewrite a reader's custom title.
export function currentReplayTitle(snapshot, savedTitle) {
  if (snapshot?.replay?.strategyVersion !== 'v0.12') return null;
  const canonical = snapshot?.title;
  if (typeof canonical !== 'string' || !/^(NQ|ES)=F · v0\.12 追体験$/u.test(canonical)) return null;
  if (typeof savedTitle !== 'string') return null;
  const prior = /^(NQ|ES)=F · v0\.(\d+) 追体験$/u.exec(savedTitle);
  return prior?.[1] === canonical.slice(0, 2) && prior?.[2] === '11' ? canonical : null;
}
