import React, {useEffect, useState} from 'react';
type API=(path:string,body?:unknown,method?:string)=>Promise<any>;
type Entry={id:string;input:{Narrative:string;EntryId:string;UserId:string};accepted_id:string|null;revision:number;state:string};
type Matter={id:string;client_name:string;matter_name:string};
const normalized=(s:string)=>s.toLowerCase().trim().replace(/\s+/g,' ');
export default function FeedbackExperiment({api,run,onComparison}:{api:API;run:any;onComparison:(c:any)=>void}){
 const [entries,setEntries]=useState<Entry[]>([]),[source,setSource]=useState<Entry|null>(null),[search,setSearch]=useState(''),[matterSearch,setMatterSearch]=useState(''),[matters,setMatters]=useState<Matter[]>([]),[target,setTarget]=useState('');
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState(''),[original,setOriginal]=useState<{entry:Entry;updated:Entry}|null>(null);
 useEffect(()=>{let active=true;const t=setTimeout(()=>{api('/time-entries?'+new URLSearchParams({timekeeper:run.input.UserId,q:search,limit:'100'})).then(v=>{if(active)setEntries(v.items.filter((e:Entry)=>e.input.EntryId!==run.input.EntryId&&normalized(e.input.Narrative)!==normalized(run.input.Narrative)))}).catch(e=>{if(active)setError(e.message)})},200);return()=>{active=false;clearTimeout(t)}},[api,run.id,search]);
 useEffect(()=>{let active=true;const t=setTimeout(()=>{api('/matters?'+new URLSearchParams({q:matterSearch,limit:'100'})).then(v=>{if(active)setMatters(v.items)}).catch(e=>{if(active)setError(e.message)})},200);return()=>{active=false;clearTimeout(t)}},[api,matterSearch]);
 const candidates: Matter[]=run.candidates.map((c:any)=>({id:c.CandidateId,client_name:c.ClientName,matter_name:c.MatterName}));
 const options=matterSearch?matters:[...candidates,...matters.filter(m=>!candidates.some(c=>c.id===m.id))];
 async function refreshProbe(){onComparison(await api('/debug/runs/'+run.id+'/compare',{}))}
 async function apply(){if(!source||!target)return;setBusy(true);setError('');setNotice('');try{
  const updated=await api('/time-entries/'+source.id+'/decisions',{revision:source.revision,action:'accept',candidate_id:target});
  setOriginal({entry:source,updated});setSource(updated);setNotice('Assignment saved. Reevaluating the fixed probe…');
  await refreshProbe();setNotice('Assignment saved and probe reevaluated. Compare the graph and scores below.');
 }catch(e){setError((e as Error).message+' If the assignment was saved, use Compare with current feedback to retry the comparison.')}finally{setBusy(false)}}
 async function restore(){if(!original)return;setBusy(true);setError('');setNotice('');try{
  const updated=await api('/time-entries/'+original.updated.id+'/decisions',{revision:original.updated.revision,action:original.entry.accepted_id?'accept':'undo',candidate_id:original.entry.accepted_id});
  setSource(updated);setOriginal(null);setNotice('Previous assignment restored. Reevaluating…');await refreshProbe();setNotice('Previous assignment restored; probe reevaluated. Review history retains both decisions.');
 }catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 return <section className="feedback-experiment"><h3>Correct another entry. Watch this probe change.</h3><p><strong>Fixed probe:</strong> {run.input.Narrative}</p><p>Choose a different narrative by <strong>{run.input.UserId}</strong>. This saves a real, reversible review decision. The probe’s assignment and the model weights stay unchanged.</p>
  <div className="feedback-fields"><label>Find another entry<input aria-label="Find feedback entry" value={search} disabled={busy||!!original} onChange={e=>{setSearch(e.target.value);setSource(null)}} placeholder="Narrative or entry ID"/><select aria-label="Feedback source entry" value={source?.id||''} disabled={busy||!!original} onChange={e=>{setSource(entries.find(x=>x.id===e.target.value)||null);setError('');setNotice('')}}><option value="">Choose another entry…</option>{entries.map(e=><option key={e.id} value={e.id}>{e.input.EntryId} · {e.input.Narrative.slice(0,100)}</option>)}</select></label>
  <label>Assign that entry to<input aria-label="Find feedback matter" value={matterSearch} disabled={busy||!!original} onChange={e=>{setMatterSearch(e.target.value);setTarget('')}} placeholder="Client, matter, or ID"/><select aria-label="Feedback target matter" value={target} disabled={busy||!!original} onChange={e=>setTarget(e.target.value)}><option value="">Choose client / matter…</option>{options.map(m=><option key={m.id} value={m.id}>{m.client_name} / {m.matter_name}</option>)}</select></label></div>
  {source&&<blockquote>{source.input.Narrative}<small>{source.input.EntryId} · {source.state}{source.accepted_id?' · Already confirmed; this changes its assignment':''}</small></blockquote>}
  {!entries.length&&<p>No other matching entries. Import another entry for this timekeeper or change the search.</p>}
  <div className="feedback-actions"><button className="hub-primary" disabled={busy||!source||!target||!!original||source.accepted_id===target} onClick={()=>void apply()}>Apply correction & reevaluate probe</button>{original&&<><button className="hub-secondary" disabled={busy} onClick={()=>void restore()}>Restore previous assignment & reevaluate</button><button className="hub-secondary" disabled={busy} onClick={()=>{setOriginal(null);setSource(null);setNotice('Assignment kept. Choose another source entry to add more feedback.')}}>Keep assignment & choose another entry</button></>}</div>
  {original&&<p>Restore this experiment, or keep the assignment and try another source. Saved assignments remain editable in Time entries.</p>}
  {notice&&<p role="status">{notice}</p>}{error&&<p role="alert" className="hub-alert">{error}</p>}
 </section>
}
