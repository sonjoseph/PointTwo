const {contextBridge,ipcRenderer}=require('electron');
contextBridge.exposeInMainWorld('pointtwo',{token:()=>ipcRenderer.invoke('pointtwo:token')});
