const {app,BrowserWindow,session,dialog,ipcMain}=require('electron');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
app.setName('PointTwo');
if(!app.requestSingleInstanceLock()){app.quit();}else{
 let backend,window;
 const root=path.resolve(__dirname,'..');
 const target='http://127.0.0.1:8794';
 const data=path.resolve(process.env.POINTTWO_DATA||path.join(root,'.pointtwo','python'));
 async function ready(){try{const r=await fetch(target+'/api/health');return r.ok&&(await r.json()).engine==='python-pgvector'}catch{return false}}
 app.whenReady().then(async()=>{
  if(app.isPackaged){dialog.showErrorBox('Development runtime required','Run this take-home application from its repository. Distribution packaging must include a configured Python and Postgres runtime.');app.quit();return;}
  if(!await ready()){
   const python=process.env.POINTTWO_PYTHON||path.join(root,'.pointtwo','classifier-venv','bin','python');
   const config=path.join(root,'.pointtwo','classification-postgres.json');
   const databaseUrl=process.env.DATABASE_URL||(fs.existsSync(config)?JSON.parse(fs.readFileSync(config,'utf8')).database_url:null);
   if(!fs.existsSync(python)||!databaseUrl){dialog.showErrorBox('Local setup required','Follow README.md to install the Python dependencies and configure Postgres.');app.quit();return;}
   fs.mkdirSync(data,{recursive:true,mode:0o700});
   const log=fs.openSync(path.join(data,'engine.log'),'a',0o600);
   backend=spawn(python,['-m','classifier'],{cwd:root,env:{...process.env,DATABASE_URL:databaseUrl,POINTTWO_DATA:data,POINTTWO_PORT:'8794'},stdio:['ignore',log,log]});fs.closeSync(log);
   backend.on('error',()=>{dialog.showErrorBox('Could not start PointTwo','See .pointtwo/python/engine.log.');app.quit()});
   for(let i=0;i<150&&!await ready();i++)await new Promise(r=>setTimeout(r,200));
  }
  if(!await ready()){dialog.showErrorBox('PointTwo could not start','Check Postgres and the local engine log.');app.quit();return;}
  window=new BrowserWindow({width:1450,height:980,minWidth:800,minHeight:600,title:'PointTwo',backgroundColor:'#f6f7f2',webPreferences:{nodeIntegration:false,contextIsolation:true,sandbox:true,preload:path.join(__dirname,'preload.cjs')}});
  ipcMain.handle('pointtwo:token',event=>{
   if(!event.senderFrame||new URL(event.senderFrame.url).origin!==target)throw new Error('Untrusted window');
   return fs.readFileSync(path.join(data,'access-token'),'utf8').trim();
  });
  session.defaultSession.webRequest.onBeforeRequest((details,callback)=>{
   const url=new URL(details.url);callback({cancel:!['data:','blob:','devtools:'].includes(url.protocol)&&url.origin!==target});
  });
  window.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  window.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==target)event.preventDefault()});
  await window.loadURL(target);
 });
 app.on('second-instance',()=>{if(window){window.show();window.focus()}});
 app.on('before-quit',()=>{if(backend&&!backend.killed)backend.kill('SIGINT')});
 app.on('window-all-closed',()=>app.quit());
}
