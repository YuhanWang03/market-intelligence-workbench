/* eslint-disable @typescript-eslint/no-require-imports */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const file = path.resolve(__dirname, '../app/relationship-map.tsx');
const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const loaded = new Module(file, module);
loaded.filename = file;
loaded.paths = module.paths;
loaded._compile(compiled, file);
const render = (ticker, relations) => renderToStaticMarkup(React.createElement(loaded.exports.RelationshipMap, { ticker, relations }));

test('empty research does not invent companies', () => {
  const html = render('TEST', []);
  assert.match(html, /本次未返回可绘制/);
  assert.doesNotMatch(html, /NVDA|MSFT|TSM/);
});
test('nodes show actual candidate descriptions instead of evidence states', () => {
  const html = render('ACME', [
    { target_company: 'SUP', relationship_type: 'SUPPLIER', evidence_status: 'NO_EVIDENCE', description: '先进制程代工' },
    { target_company: 'CLIENT', relationship_type: 'customer', evidence_status: 'CO_MENTION', description: '训练集群采购' },
    { target_company: 'PEER', relationship_type: 'COMPETITOR', evidence_status: 'EVIDENCE_FOUND', description: '定制加速器' },
    { target_company: 'ALLY', relationship_type: 'partner' },
  ]);
  for (const value of ['ACME', 'SUP', 'CLIENT', 'PEER', 'ALLY', '先进制程代工', '训练集群采购', '定制加速器', '暂无候选说明', '其他关系']) assert.ok(html.includes(value));
  assert.doesNotMatch(html, /未找到关系证据|仅搜索共同提及|发现关系证据/);
  assert.equal((html.match(/class="relation-map-node(?:\s|")/g) || []).length, 4);
  assert.doesNotMatch(html, /NVDA/);
});
test('escapes upstream text', () => {
  assert.ok(render('TEST', [{ target_company: '<script>alert(1)</script>', relationship_type: 'customer' }]).includes('&lt;script&gt;'));
});
test('graph is between summary and relationship cards', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '../app/page.tsx'), 'utf8');
  const start = source.indexOf('function SupplyChainView');
  const graph = source.indexOf('<RelationshipMap ticker={result.ticker}', start);
  assert.ok(graph > source.indexOf('{result.ticker} 产业关系', start));
  assert.ok(graph < source.indexOf('<h2>公司产业关系</h2>', start));
});
