// Paste into the browser dev-tools console on
// https://www.ticketdata.com/performer/mississippi-state-bulldogs-football?tab=past  (or ?tab=upcoming)
// Prints CSV rows in data/tickets.csv order: opponent,date,getin,observed
// Reads only the already-loaded DOM; makes no network requests.
(() => {
  const today = new Date().toISOString().slice(0, 10);
  const seen = new Set();
  const lines = ["opponent,date,getin,observed"];
  for (const r of document.querySelectorAll("table tbody tr")) {
    const c = [...r.querySelectorAll("td")].map((td) => td.innerText.trim());
    if (c.length < 8) continue;
    const ev = c[1]
      .replace(/^Egg Bowl - /, "")
      .replace(/ \(Rescheduled.*\)$/, "")
      .replace(/ at Mississippi State Bulldogs( Football)?$/, "");
    const m = c[2].match(/(\d{2})\/(\d{2})\/(\d{4})/);
    if (!m) continue;
    const date = `${m[3]}-${m[1]}-${m[2]}`;
    const key = ev + "|" + date;
    if (seen.has(key)) continue;
    seen.add(key);
    const price = (c[7].match(/\$(\d+(?:\.\d+)?)/) || [])[1] || "";
    const opp = ev.includes(",") ? `"${ev}"` : ev;
    lines.push([opp, date, price, today].join(","));
  }
  console.log(lines.join("\n"));
  return `${lines.length - 1} rows`;
})();
