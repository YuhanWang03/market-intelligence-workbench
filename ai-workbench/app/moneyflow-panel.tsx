'use client';

type Reading = { price: number; price_return: number; price_state: string; cmf: number; flow_state: string; rsi: number; rsi_zone: string; rsi_divergence: string };
export type FlowAnalysis = { status: string; reason: string; reading?: Reading | null; signal?: { kind: string; strength: string; bull: string; bear: string } | null; chart_png?: string | null; chart_status: string; narration_status: string; as_of?: string; bar_count: number; data_note: string; config: { price_window: number; cmf_window: number; rsi_window: number; cmf_inflow_threshold: number; cmf_outflow_threshold: number; flat_band: number }; llm_tokens?: number };
const labels: Record<string, string> = { up: '上涨', flat: '横盘', down: '下跌', inflow: '流入倾向', outflow: '流出倾向', neutral: '中性', oversold: '超卖', low: '低位', high: '高位', overbought: '超买', bullish: '看涨背离', bearish: '看跌背离', none: '未发现动量背离', strong: '较强', moderate: '中等' };
const number = (value: number, percent = false) => Number.isFinite(value) ? `${(value * (percent ? 100 : 1)).toFixed(2)}${percent ? '%' : ''}` : '—';

export function MoneyflowPanel({ analysis, ticker }: { analysis?: FlowAnalysis; ticker: string }) {
  const reading = analysis?.reading;
  const signal = analysis?.signal;
  const verdict = !analysis ? '待重新研究' : analysis.status === 'SIGNAL' && signal ? `${signal.kind === 'accumulation' ? '疑似吸筹' : '疑似派发'} · ${labels[signal.strength] || signal.strength}` : analysis.status === 'NO_SIGNAL' ? '未触发背离' : '数据不足或无效';
  return <section className="surface engine-card flow-depth">
    <div className="surface-header"><div><h2>量价背离分析 · /flow</h2><span>价格、CMF 与 RSI 联合判断；信号由规则计算，多空解读由 AI 补充</span></div><span className="engine-status">{verdict}</span></div>
    {!analysis ? <p className="insufficient-note">这份旧研究尚不包含 /flow 分析，请重新运行资金流模块。</p> : <>
      <p className="flow-method">{analysis.reason} 数据截至：{analysis.as_of || '未知'} · {analysis.bar_count} 个交易日。</p>
      {reading && <div className="flow-axis-grid">
        <article><h3>价格轴 · {labels[reading.price_state]}</h3><strong>{number(reading.price_return, true)}</strong><p>近 {analysis.config.price_window} 个交易日涨跌幅 · 收盘价 {number(reading.price)}</p></article>
        <article><h3>资金流轴 · {labels[reading.flow_state]}</h3><strong>{number(reading.cmf)}</strong><p>CMF{analysis.config.cmf_window} · 成交量加权代理指标</p></article>
        <article><h3>动量轴 · {labels[reading.rsi_zone]}</h3><strong>{number(reading.rsi)}</strong><p>RSI{analysis.config.rsi_window} · {labels[reading.rsi_divergence]}</p></article>
      </div>}
      {analysis.chart_png ? <figure className="flow-chart">
        {/* Existing /flow renderer returns a PNG, not an external image URL. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`data:image/png;base64,${analysis.chart_png}`} alt={`${ticker} 日线价格、CMF 资金流代理和 RSI 动量三联图`}/>
        <figcaption>上：收盘价格（Price） · 中：CMF · 下：RSI；三图共享交易日轴，展示最近至多 60 个交易日。</figcaption>
      </figure> : reading && <p className="insufficient-note">三联图暂不可用，指标与背离判断仍可查看。</p>}
      {signal && <div className="flow-axis-grid flow-narration"><article><h3>多头解读</h3><p>{signal.bull || '本次 AI 解读未取得，规则信号不受影响。'}</p></article><article><h3>空头解读</h3><p>{signal.bear || '本次 AI 解读未取得，规则信号不受影响。'}</p></article></div>}
      {analysis.status === 'NO_SIGNAL' && <p className="flow-method">未触发背离不代表没有资金活动；本次不调用 AI 生成背离解读。</p>}
      <details className="flow-rules"><summary>数据口径与判定规则</summary><p>{analysis.data_note}</p><p>价格横盘或下跌且 CMF 流入：检查潜在吸筹；价格横盘或上涨且 CMF 流出：检查潜在派发。RSI 位置与背离用于筛选和分级。</p><p>CMF 流入阈值 ≥ {analysis.config.cmf_inflow_threshold}，流出阈值 ≤ {analysis.config.cmf_outflow_threshold}；价格横盘区间 ±{number(analysis.config.flat_band, true)}。这不是逐笔成交或真实机构账户资金流，不能据此确认主力行为。</p></details>
    </>}
  </section>;
}
