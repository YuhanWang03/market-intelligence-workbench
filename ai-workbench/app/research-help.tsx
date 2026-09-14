'use client';

import { useId, useRef } from 'react';
import { RESEARCH_HELP, BILLING_GUIDE } from './research-help-content';
import './research-help.css';

export function ResearchHelp({ tool }: { tool: keyof typeof RESEARCH_HELP }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const id = useId();
  const help = RESEARCH_HELP[tool];
  return <>
    <button ref={trigger} type="button" className="research-info-button" aria-label={`了解${help.title}`} aria-haspopup="dialog" onClick={() => dialog.current?.showModal()}><span aria-hidden="true">i</span></button>
    <dialog ref={dialog} className="research-help-dialog" aria-labelledby={id} onClose={() => trigger.current?.focus()} onClick={event => {
      if (event.target !== event.currentTarget) return;
      const rect = event.currentTarget.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.currentTarget.close();
    }}>
      <header><div><span>研究工具说明</span><h2 id={id}>{help.title}</h2></div><button type="button" autoFocus aria-label="关闭说明" onClick={() => dialog.current?.close()}>×</button></header>
      <div className="research-help-content">
        <p className="research-help-intro">{help.intro}</p>
        {help.features.map(([heading, text], index) => <section key={heading}><h3><span aria-hidden="true">0{index + 1}</span>{heading}</h3><p>{text}</p></section>)}
        <aside><strong>使用这页，大致会有哪些花费？</strong><p>{help.cost}</p></aside>
        <details className="research-help-billing"><summary>展开计费口径、算例与缓存说明</summary>{BILLING_GUIDE.map(([title, text]) => <section key={title}><h3>{title}</h3><p>{text}</p></section>)}</details>
      </div>
      <footer>以上是功能与计费方式说明，不是本次查询的账单。实际来源、时间和缺失项请看本页数据质量；实际已记录调用和采用单价请在“花费”页核对。打开说明不会产生 API 费用。</footer>
    </dialog>
  </>;
}
