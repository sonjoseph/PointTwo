import React, {useEffect, useState} from 'react';

type Stage={label:string;matter_id:string|null;score?:number;rank?:number;timekeeper?:string};
type Node={id:string;kind:'query'|'catalog'|'example';text:string;x:number;y:number;before:Stage|null;after:Stage|null;cosine_similarity:number;cosine_distance:number};
type Graph={nodes:Node[];explained_variance:number;missing_points:number;baseline_candidate_count:number};
type API=(path:string,body?:unknown,method?:string)=>Promise<any>;
const palette=['#39745c','#557fb1','#aa647c','#ae7b31','#7f6bad','#37888e','#a3553e','#717e33','#b26d9c','#516766','#8e674c','#49739b','#9c943b','#5e8b78','#7d6672','#737f4c'];
export default function EmbeddingGraph({api,runId,afterId,onSelectMatter}:{api:API;runId:string;afterId?:string;onSelectMatter:(id:string)=>void}){
 const [graph,setGraph]=useState<Graph|null>(null),[error,setError]=useState(''),[phase,setPhase]=useState<'before'|'after'>('before'),[selected,setSelected]=useState('query');
 useEffect(()=>{let active=true;setGraph(null);setError('');api('/debug/runs/'+runId+'/projection'+(afterId?'?after='+encodeURIComponent(afterId):'')).then(g=>{if(active){setGraph(g);setPhase(afterId?'after':'before');setSelected('query')}}).catch(e=>{if(active)setError(e.message)});return()=>{active=false}},[api,runId,afterId]);
 const nodes=graph?.nodes||[],visible=nodes.filter(n=>n[phase]);
 const matterIds=[...new Set(nodes.flatMap(n=>[n.before?.matter_id,n.after?.matter_id]).filter(Boolean))];
 const color=(id:string|null)=>id?palette[Math.max(0,matterIds.indexOf(id))%palette.length]:'#243a31';
 const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y);
 const minX=Math.min(0,...xs),maxX=Math.max(0,...xs),minY=Math.min(0,...ys),maxY=Math.max(0,...ys);
 const span=Math.max(maxX-minX,maxY-minY,.1),cx=(minX+maxX)/2,cy=(minY+maxY)/2;
 // Equal units on both axes, with shared bounds for before and after.
 const x=(n:Node)=>370+(n.x-cx)*300/span,y=(n:Node)=>190-(n.y-cy)*300/span;
 const labelPositions:{id:string;x:number;y:number}[]=[];
 for(const n of visible.filter(n=>n.kind!=='example')){let ly=y(n)-12;const lx=x(n)+12;while(labelPositions.some(p=>Math.abs(p.x-lx)<55&&Math.abs(p.y-ly)<16))ly+=17;labelPositions.push({id:n.id,x:lx,y:ly})}
 const chosen=visible.find(n=>n.id===selected)||visible.find(n=>n.kind==='query');
 function choose(n:Node){setSelected(n.id);const id=n[phase]?.matter_id;if(id)onSelectMatter(id)}
 return <section className="embedding-panel" aria-label="Embedding neighborhood">
  <div className="embedding-heading"><div><h3>Embedding neighborhood</h3><p>The entry, nearby client–matter descriptions, and confirmed examples.</p></div>{afterId&&<div className="embedding-toggle" aria-label="Graph comparison"><button aria-pressed={phase==='before'} onClick={()=>setPhase('before')}>Baseline</button><button aria-pressed={phase==='after'} onClick={()=>setPhase('after')}>After feedback</button></div>}</div>
  {error?<p role="alert">{error}</p>:!graph?<p role="status">Loading saved embedding evidence…</p>:<>
   <div className="embedding-legend"><span>◆ Entry being classified</span><span>■ Client–matter description</span><span>● Confirmed time entry</span></div>
   <svg className="embedding-plot" viewBox="0 0 740 380" aria-label={`${phase==='before'?'Baseline':'After feedback'} embedding projection. Select a point to inspect exact cosine distance.`}>
    <line x1="25" x2="715" y1="190" y2="190" stroke="#e1e8dc"/><line x1="370" x2="370" y1="20" y2="355" stroke="#e1e8dc"/>
    <text x="710" y="372" textAnchor="end" className="embedding-axis">PCA component 1</text><text x="12" y="16" className="embedding-axis">PCA component 2</text>
    {visible.filter(n=>n.kind==='example').map(n=>{const catalog=visible.find(c=>c.kind==='catalog'&&c[phase]?.matter_id===n[phase]?.matter_id);return catalog?<line key={'edge'+n.id} x1={x(n)} y1={y(n)} x2={x(catalog)} y2={y(catalog)} stroke={color(n[phase]!.matter_id)} strokeOpacity=".35" strokeDasharray="4 4"/>:null})}
    {[...visible].sort((a,b)=>(a.kind==='query'?1:0)-(b.kind==='query'?1:0)).map(n=><g key={n.id} tabIndex={0} role="button" aria-label={`${n.kind}: ${n[phase]!.label}; cosine distance ${n.cosine_distance.toFixed(4)}`} onClick={()=>choose(n)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(n)}}} className="embedding-node" transform={`translate(${x(n)},${y(n)})`}>
     <title>{n[phase]!.label+'\n'+n.text+'\nCosine distance: '+n.cosine_distance.toFixed(4)}</title><circle r="15" fill="transparent" stroke={chosen?.id===n.id?'#243a31':'transparent'} strokeWidth="2"/>
     {n.kind==='query'?<path d="M 0 -10 L 10 0 L 0 10 L -10 0 Z" fill="#243a31"/>:n.kind==='catalog'?<rect x="-6" y="-6" width="12" height="12" rx="2" fill={color(n[phase]!.matter_id)}/>:<circle r="5" fill={color(n[phase]!.matter_id)}/>}
     {(n.kind==='query'||n.kind==='catalog')&&<text x="12" y={(labelPositions.find(p=>p.id===n.id)?.y||y(n)-12)-y(n)} className="embedding-label">{n.kind==='query'?'Your entry':`#${n[phase]!.rank}`}</text>}
    </g>)}
   </svg>
   <div className="embedding-details" aria-live="polite">{chosen&&<><strong>{chosen[phase]!.label}</strong><p>{chosen.text}</p><div><span>Cosine similarity <b>{chosen.cosine_similarity.toFixed(4)}</b></span><span>Cosine distance <b>{chosen.cosine_distance.toFixed(4)}</b></span>{chosen[phase]?.score!==undefined&&<span>Ranking score <b>{chosen[phase]!.score!.toFixed(4)}</b></span>}</div>{chosen.before&&chosen.after&&chosen.before.matter_id!==chosen.after.matter_id&&<p>Assignment changed: {chosen.before.label} → {chosen.after.label}</p>}</>}</div>
   <details className="embedding-key"><summary>Client–matter color key ({visible.filter(n=>n.kind==='catalog').length})</summary><div>{visible.filter(n=>n.kind==='catalog').map(n=><button key={n.id} onClick={()=>choose(n)}><i style={{background:color(n[phase]!.matter_id)}}/>#{n[phase]!.rank} {n[phase]!.label}</button>)}</div></details>
   <p className="debug-caption">Fixed baseline PCA axes · {(graph.explained_variance*100).toFixed(1)}% of baseline variance shown. The map is a 2D approximation; cosine values use all 1,024 dimensions. Shows up to 8 ranked matters and 3 retrieved examples per matter. Lines represent confirmed assignments, not HNSW edges.</p>
   {afterId&&<p className="debug-caption">Feedback changes labels and available evidence; it does not move unchanged embeddings. A missing point may simply fall outside this run’s retrieved neighborhood.</p>}
   {!!graph.missing_points&&<p role="status">{graph.missing_points} historical evidence vector(s) unavailable and omitted.</p>}
  </>}
 </section>
}
