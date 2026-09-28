// Checks the page's JS SIP simulation against backtest50_report.sip_sim (Python).
//   node tests/test_sip.js
const fs = require('fs');
const html = fs.readFileSync(__dirname + '/../docs/backtest_50x.html', 'utf8');
const start = html.indexOf('const D = ') + 'const D = '.length;
const D = JSON.parse(html.slice(start, html.indexOf('\n', start)).trim().replace(/;$/, ''));
const code = html.split('/*SIM-START*/')[1].split('/*SIM-END*/')[0];
const api = new Function(code + '\nreturn {simulate, xirr};')();
const end = Date.parse(D.meta.end + 'T23:59:00+05:30');
for (const [p, from] of [['2021', '2021-01-01'], ['2024', '2024-01-01']])
  for (const acc of ['ml', 'base'])
    for (const sip of [0, 500]) {
      const r = api.simulate(D.periods[p][acc].sip, from, sip);
      console.log(`from ${p} ${acc.padEnd(4)} sip ${String(sip).padStart(3)}: invested ${r.invested} -> Rs ${r.final.toFixed(0)}` +
                  ` | XIRR ${api.xirr(r.flows, r.final, end).toFixed(1)}% | worst fall ${r.dd.toFixed(1)}%`);
    }
