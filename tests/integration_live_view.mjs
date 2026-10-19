// Isolated headless application test. Does not attach to the user's browser.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const fixtureDir = await mkdtemp(path.join(import.meta.dirname, '.qa-'));
const profile = await mkdtemp(path.join(tmpdir(), '3dgs-live-qa-'));
await mkdir(fixtureDir, { recursive: true });
const header = `ply\nformat binary_little_endian 1.0\nelement vertex 3\n${['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'].map(name => `property float ${name}\n`).join('')}end_header\n`;
const vertices = [
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 12, -2, -2, -2, 1, 0, 0, 0],
    [0, 0, .8, 0, 0, 0, 0, 1, 0, 14, -1.5, -1.5, -1.5, 1, 0, 0, 0],
    [.45, 0, 0, 0, 0, 0, 1, 0, 0, 12, -2.5, -2.5, -2.5, 1, 0, 0, 0]
].flat();
const bytes = Buffer.alloc(vertices.length * 4);
vertices.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
await writeFile(path.join(fixtureDir, 'visibility.ply'), Buffer.concat([Buffer.from(header), bytes]));
const chrome = spawn(process.env.CHROME_BIN ?? '/usr/bin/google-chrome', [
    '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--window-size=1440,1080', '--no-first-run', 'about:blank'
], { stdio: ['ignore', 'ignore', 'pipe'] });
let browserLog = '';
chrome.stderr.on('data', chunk => { browserLog = (browserLog + chunk).slice(-5000); });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let cdp;
const connect = async url => {
    const ws = new WebSocket(url);
    await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }); });
    let id = 0;
    const pending = new Map();
    const errors = [];
    ws.addEventListener('message', event => {
        const message = JSON.parse(event.data);
        if (pending.has(message.id)) {
            const { resolve, reject } = pending.get(message.id);
            pending.delete(message.id);
            message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
        } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
        else if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') errors.push(message.params.entry);
    });
    const send = (method, params = {}) => new Promise((resolve, reject) => {
        pending.set(++id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
    });
    return { ws, send, errors };
};
const evaluate = async expression => {
    const result = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
};
const unexpectedBrowserErrors = () => cdp.errors.filter(error =>
    !(error.url?.endsWith('/api/depth') && error.text?.includes('429 (Too Many Requests)')));
const waitFor = async (expression, timeout = 90000) => {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        const result = await evaluate(expression);
        if (result) return result;
        await delay(300);
    }
    throw new Error(`Timed out: ${expression}\n${JSON.stringify(await evaluate('document.body.innerText.slice(-2500)'))}\n${JSON.stringify(cdp.errors)}`);
};

try {
    let port;
    for (let attempt = 0; attempt < 60; attempt++) {
        try { port = Number((await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); break; } catch { await delay(200); }
    }
    if (!port) throw new Error(`Headless Chrome did not start: ${browserLog}`);
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    cdp = await connect(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await cdp.send('Runtime.enable');
    await cdp.send('Log.enable');
    await cdp.send('Page.enable');
    const syntheticUrl = `http://127.0.0.1:8027/3dgs-splat-labeler/app/?load=../tests/${path.basename(fixtureDir)}/visibility.ply&filename=visibility.ply`;
    await cdp.send('Page.navigate', { url: syntheticUrl });
    await waitFor('window.scene && window.scene.events.functions.has("annotation.panel") && window.scene.getElementsByType("splat").length');
    // Set annotations and poses through the application API, without UI automation.
    const configured = await evaluate(`(() => {
        const scene=window.scene, panel=scene.events.invoke('annotation.panel'), splat=panel.activeSplat();
        const p=scene.camera.mainCamera.getPosition().clone();
        const points=[0,1,2].map(index=>{const point=p.clone();splat.calcSplatWorldPosition(index,point);return point;}).sort((a,b)=>a.lengthSq()-b.lengthSq());
        const [bg,side,fg]=points;
        const direction=fg.clone().sub(bg).normalize();
        const position=bg.clone().add(direction.clone().mulScalar(3));
        scene.events.fire('camera.setPose',{position,target:bg.clone()},0);
        const center=bg.clone().add(side).mulScalar(.5);
        const size=[Math.abs(side.x-bg.x)+.12,Math.abs(side.y-bg.y)+.12,Math.abs(side.z-bg.z)+.12];
        panel.boxes=[{id:'visible_test',label:'branch',operation:'include',center:[center.x,center.y,center.z],size,euler:[0,0,0]}];
        panel.activeBoxIndex=0; panel.box=panel.boxes[0]; panel.touched=true; panel.writeInputs();
        scene.forceRender=true;
        window._qaPoints={bg:[bg.x,bg.y,bg.z],side:[side.x,side.y,side.z]};
        return {bg:window._qaPoints.bg,fg:[fg.x,fg.y,fg.z],side:window._qaPoints.side};
    })()`);
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && window.scene.events.invoke("annotation.liveSnapshot")?.pose.segmentation.annotations[0]?.id === "visible_test"');
    const occlusion = await evaluate(`(() => {
        const scene=window.scene, result=scene.events.invoke('annotation.liveSnapshot');
        const at=(coordinates)=>{
            const p=scene.camera.mainCamera.getPosition().clone().set(...coordinates);
            const uv=p.clone(); scene.camera.worldToScreen(p,uv);
            const x=Math.min(639,Math.max(0,Math.round(uv.x*640))), y=Math.min(639,Math.max(0,Math.round(uv.y*640)));
            return {x,y,value:result.mask.getContext('2d').getImageData(x,y,1,1).data[0]};
        };
        return {behind:at(window._qaPoints.bg),visible:at(window._qaPoints.side),pose:result.pose,depth:result.depth?.length,frame:result.frameId};
    })()`);
    console.log(JSON.stringify({configured, behind:occlusion.behind, visible:occlusion.visible, frame:occlusion.frame}));
    assert.equal(occlusion.behind.value, 0, 'foreground must occlude the annotated background');
    assert.equal(occlusion.visible.value, 255, 'unoccluded annotated branch must remain in the mask');
    assert.equal(occlusion.depth, 640 * 640);
    assert.equal(occlusion.pose.frame_id, occlusion.frame);
    assert.equal(await evaluate('scene.events.invoke("annotation.liveSnapshot").detection.frameId'), occlusion.frame);
    assert.ok(Math.abs(occlusion.pose.camera_profile.horizontal_fov_degrees - 118) < 1e-9);
    const oriented = await evaluate(`(async () => {
        const live=scene.events.invoke('annotation.panel').liveView;
        const snapshot=scene.events.invoke('annotation.liveSnapshot');
        const mask=document.createElement('canvas');mask.width=mask.height=640;
        const ctx=mask.getContext('2d');ctx.fillStyle='white';
        ctx.fillRect(308,310,24,240);ctx.fillRect(120,306,400,9);
        ctx.strokeStyle='white';ctx.lineWidth=8;ctx.beginPath();ctx.moveTo(80,80);ctx.lineTo(240,240);ctx.stroke();
        const fake={...snapshot,frameId:'qa_horizontal_trunk',mask,depth:new Float32Array(640*640).fill(.5),
            pose:{...snapshot.pose,world_up_axis:'z',camera_to_world:[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]}};
        fake.detection=await live.detectBranches(fake);
        const raw=live.branchView(fake);
        live.landingTopK=1;const topOne=live.branchView(fake);
        live.landingTopK=3;
        live.horizontalAngle=45;const wideAngle=live.branchView(fake);live.horizontalAngle=25;
        live.excludeTrunk=true;const filtered=live.branchView(fake);live.excludeTrunk=false;
        live.viewMode='candidates';
        live.drawBranches(fake);const list=live.branchList.textContent;
        live.viewMode='raw';live.drawBranches(snapshot);
        return {count:raw.branches.length,trunks:raw.branches.filter(b=>b.is_trunk).map(b=>b.id),
            horizontal:raw.horizontal.map(b=>b.id),wideHorizontal:wideAngle.horizontal.map(b=>b.id),
            topOne:topOne.landing.map(b=>b.id),topOneMaskIds:[...new Set(topOne.landingMask)].filter(Boolean),
            filteredIds:[...new Set(filtered.instanceMask)].filter(Boolean),
            excluded:filtered.excludedTrunkIds,list};
    })()`);
    assert.ok(oriented.count >= 3 && oriented.trunks.length >= 1 && oriented.horizontal.length >= 1);
    assert.ok(oriented.wideHorizontal.length > oriented.horizontal.length, 'the angle control must include the inclined branch');
    assert.equal(oriented.topOne.length,1);
    assert.deepEqual(oriented.topOneMaskIds,oriented.topOne,'the landing overlay must only contain the top-ranked candidate');
    assert.ok(oriented.excluded.every(id => !oriented.filteredIds.includes(id)));
    assert.match(oriented.list,/Landing top 3/);
    console.log(JSON.stringify({ oriented }));

    // Hold one depth response, move the camera, then release the old response.
    await evaluate(`(() => {
        const actualFetch=window.fetch.bind(window);
        window._qaOriginalFetch=actualFetch;
        window._qaHeldFrame=null;
        window.fetch=async (...args)=>{
            if(args[0]==='/api/depth' && !window._qaHeldFrame){
                window._qaHeldFrame=JSON.parse(args[1].body).frame_id;
                await new Promise(resolve=>window._qaRelease=resolve);
            }
            return actualFetch(...args);
        };
        scene.events.invoke('annotation.panel').liveView.refresh();
        return true;
    })()`);
    await waitFor('window._qaHeldFrame');
    const movement = await evaluate(`(() => {
        const position=scene.camera.mainCamera.getPosition().clone(); position.x+=.08;
        const target=position.clone().set(...window._qaPoints.bg);
        scene.events.fire('camera.setPose',{position,target},0);
        window._qaExpectedPosition=[position.x,position.y,position.z];
        scene.forceRender=true;
        return {oldFrame:window._qaHeldFrame,position:window._qaExpectedPosition};
    })()`);
    await delay(600);
    assert.equal(await evaluate('document.querySelector(".live-save button").disabled'), true);
    await evaluate('window._qaRelease()');
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").frameId !== window._qaHeldFrame');
    const moved = await evaluate('scene.events.invoke("annotation.liveSnapshot").pose');
    assert.notEqual(moved.frame_id, movement.oldFrame);
    for (let i=0;i<3;i++) assert.ok(Math.abs(moved.position[i]-movement.position[i])<1e-4, 'depth must match the moved pose');
    await evaluate('window.fetch=window._qaOriginalFetch');
    console.log(JSON.stringify({staleFrameDiscarded:movement.oldFrame,newFrame:moved.frame_id,newPosition:moved.position}));

    // Annotation edits invalidate membership even when the camera stays still.
    await evaluate(`(() => {
        const panel=scene.events.invoke('annotation.panel');
        window._qaOriginalBoxes=JSON.parse(JSON.stringify(panel.boxes));
        panel.pushBoxUndo();
        panel.boxes.push({id:'remove_test',label:'remove',operation:'exclude',center:window._qaPoints.side,size:[.2,.2,.2],euler:[0,0,0]});
        panel.syncBoxManager(); scene.forceRender=true;
        return true;
    })()`);
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").pose.segmentation.annotations.length === 2');
    const removedArea = await evaluate(`(()=>{const m=scene.events.invoke('annotation.liveSnapshot').mask.getContext('2d').getImageData(0,0,640,640).data;let n=0;for(let i=0;i<m.length;i+=4)if(m[i])n++;return n;})()`);
    assert.equal(removedArea,0,'remove BBox must remove the visible branch');
    await evaluate('scene.events.invoke("annotation.panel").undoBoxStep()');
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").pose.segmentation.annotations.length === 1');
    const restoredArea = await evaluate(`(()=>{const m=scene.events.invoke('annotation.liveSnapshot').mask.getContext('2d').getImageData(0,0,640,640).data;let n=0;for(let i=0;i<m.length;i+=4)if(m[i])n++;return n;})()`);
    assert.ok(restoredArea>0,'Undo must restore visible segmentation');
    await evaluate(`(()=>{const panel=scene.events.invoke('annotation.panel');panel.boxes=[];panel.syncBoxManager();scene.forceRender=true;return true;})()`);
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").pose.segmentation.annotations.length === 0');
    assert.equal(await evaluate(`(()=>{const m=scene.events.invoke('annotation.liveSnapshot').mask.getContext('2d').getImageData(0,0,640,640).data;for(let i=0;i<m.length;i+=4)if(m[i])return false;return true;})()`),true,'no BBoxes must produce an empty mask');
    await evaluate(`(()=>{const panel=scene.events.invoke('annotation.panel');panel.boxes=window._qaOriginalBoxes;panel.box=panel.boxes[0];panel.activeBoxIndex=0;panel.writeInputs();scene.forceRender=true;return true;})()`);
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").pose.segmentation.annotations.length === 1');
    console.log(JSON.stringify({removeBBoxArea:removedArea,undoArea:restoredArea,emptyMaskVerified:true}));

    const beforeTransform = await evaluate(`(()=>{
        const panel=scene.events.invoke('annotation.panel'),splat=panel.activeSplat();
        window._qaSplatPosition=splat.entity.getLocalPosition().clone();
        const position=window._qaSplatPosition.clone();position.x+=1;
        const before=scene.events.invoke('annotation.liveSnapshot').frameId;
        splat.entity.setLocalPosition(position);scene.boundDirty=true;scene.forceRender=true;
        return before;
    })()`);
    await waitFor(`document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").frameId !== ${JSON.stringify(beforeTransform)}`);
    assert.equal(await evaluate(`(()=>{const m=scene.events.invoke('annotation.liveSnapshot').mask.getContext('2d').getImageData(0,0,640,640).data;for(let i=0;i<m.length;i+=4)if(m[i])return false;return true;})()`),true,'moving the model outside the BBoxes must invalidate annotation membership');
    const translatedFrame=await evaluate(`(()=>{const before=scene.events.invoke('annotation.liveSnapshot').frameId;scene.events.invoke('annotation.panel').activeSplat().entity.setLocalPosition(window._qaSplatPosition);scene.boundDirty=true;scene.forceRender=true;return before;})()`);
    await waitFor(`document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot").frameId !== ${JSON.stringify(translatedFrame)}`);
    console.log(JSON.stringify({modelTransformInvalidationVerified:true}));

    // The actual file-writer code writes into in-memory directory handles.
    await evaluate(`(() => {
        window._qaFiles=[];
        const directory=(name)=>({name,getDirectoryHandle:async(child)=>directory(child),getFileHandle:async(file)=>({createWritable:async()=>({write:async(blob)=>{
            const bytes=new Uint8Array(await blob.arrayBuffer()); let text='';
            for(let start=0;start<bytes.length;start+=8192) text+=String.fromCharCode(...bytes.subarray(start,start+8192));
            window._qaFiles.push({path:name+'/'+file,base64:btoa(text)});
        },close:async()=>{}})})});
        scene.events.invoke('annotation.panel').outputDirHandle=directory('root');
        return true;
    })()`);
    await evaluate(`window.scene.events.invoke('annotation.panel').liveView.save()`);
    await waitFor('window._qaFiles.length === 7');
    const files = await evaluate('window._qaFiles');
    for (const file of files) {
        const filename = path.join('/tmp/annotation_live_export_qa', file.path);
        await mkdir(path.dirname(filename), { recursive: true });
        await writeFile(filename, Buffer.from(file.base64, 'base64'));
    }
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_synthetic_qa.png', Buffer.from(shot.data, 'base64'));
    console.log(JSON.stringify({ synthetic: configured, occlusion:{behind:occlusion.behind,visible:occlusion.visible}, exportFiles: files.map(file => file.path), errors: cdp.errors }, null, 2));

    // The Nerfstudio scene is Z-up: horizontal mouse look must yaw around Z.
    await cdp.send('Page.navigate', { url: `${syntheticUrl}&control=fly&up=z` });
    await waitFor('window.scene?.camera?.upAxis === "z" && window.scene.getElementsByType("splat").length');
    const beforeYaw = await evaluate(`(() => {
        const camera=scene.camera;
        const V=camera.mainCamera.getPosition().constructor;
        camera.setPose(new V(0,-3,0),new V(0,0,0),0);
        const rect=document.getElementById('canvas-container').getBoundingClientRect();
        return {azim:camera.azim,x:Math.round(rect.left+rect.width/2),y:Math.round(rect.top+rect.height/2)};
    })()`);
    await delay(250);
    await cdp.send('Input.dispatchMouseEvent', { type:'mousePressed',x:beforeYaw.x,y:beforeYaw.y,button:'left',buttons:1,clickCount:1 });
    await cdp.send('Input.dispatchMouseEvent', { type:'mouseMoved',x:beforeYaw.x+100,y:beforeYaw.y,button:'left',buttons:1 });
    await cdp.send('Input.dispatchMouseEvent', { type:'mouseReleased',x:beforeYaw.x+100,y:beforeYaw.y,button:'left',buttons:0,clickCount:1 });
    await delay(1500);
    const yawResult = await evaluate(`(() => {
        const camera=scene.camera, transform=camera.mainCamera.getWorldTransform();
        const p=camera.mainCamera.getPosition(), up=transform.getY(), forward=transform.getZ();
        return {azim:camera.azim,position:[p.x,p.y,p.z],up:[up.x,up.y,up.z],forward:[forward.x,forward.y,forward.z]};
    })()`);
    console.log(JSON.stringify({ zUpMouseYaw: { before:beforeYaw.azim, after:yawResult } }));
    assert.ok(Math.abs(yawResult.azim-beforeYaw.azim)>10, 'left drag should change yaw');
    assert.ok(yawResult.up[2]>0.99, 'camera up should stay aligned with scene Z during yaw');
    assert.ok(Math.hypot(...yawResult.position.map((value,index)=>value-[0,-3,0][index]))<0.01, 'fly yaw should keep the camera position');
    assert.ok(Math.abs(yawResult.forward[2])<0.01, 'horizontal yaw should not add pitch');

    // Load the user's actual DJI Gaussian model and run the same production API.
    const djiUrl = 'http://127.0.0.1:8027/3dgs-splat-labeler/app/?load=../../exports/dji_0967_supersplat/dji_0967_splat.ply&filename=dji_0967_splat.ply&metricScale=../../data/dji_0967_3dgs/metric_scale.json&control=fly';
    await cdp.send('Page.navigate', { url: djiUrl });
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready"', 240000);
    await evaluate(`(async () => {
        const panel=scene.events.invoke('annotation.panel');
        const picked=await scene.camera.intersect(.5,.72);
        if(!picked) throw new Error('No visible DJI surface at the test pixel');
        const point=picked.position;
        panel.boxes=[{id:'dji_live_branch',label:'branch',operation:'include',center:[point.x,point.y,point.z],size:[.6,.6,.6],euler:[0,0,0]}];
        panel.activeBoxIndex=0; panel.box=panel.boxes[0]; panel.touched=true;panel.writeInputs();
        scene.forceRender=true;
        return true;
    })()`);
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot")?.pose.segmentation.annotations[0]?.id === "dji_live_branch"', 90000);
    const dji = await evaluate(`(() => {const r=scene.events.invoke('annotation.liveSnapshot');const m=r.mask.getContext('2d').getImageData(0,0,640,640).data;let area=0;for(let i=0;i<m.length;i+=4)if(m[i])area++;const up=scene.camera.mainCamera.getWorldTransform().getY();const ids=[...new Set(r.detection.instanceMask)].filter(id=>id>0);return {frame:r.frameId,depthShape:[r.pose.width,r.pose.height],depthCount:r.depth.length,maskArea:area,branchCount:r.detection.branches.length,branchIds:ids,branchCpuMs:r.detection.cpu_ms,branchFrame:r.detection.frameId,pose:r.pose.camera_to_world,intrinsics:r.pose.intrinsics,status:document.querySelector('.live-status').textContent,splats:scene.events.invoke('annotation.panel').activeSplat().numSplats,cameraUp:[up.x,up.y,up.z]};})()`);
    const djiShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_qa.png', Buffer.from(djiShot.data, 'base64'));
    console.log(JSON.stringify({ dji, errors: cdp.errors }, null, 2));
    assert.equal(dji.depthCount, 640 * 640);
    assert.ok(dji.splats > 1000);
    assert.ok(dji.maskArea > 0, 'the actual annotated DJI surface must appear in the live mask');
    assert.ok(dji.branchCount > 0 && dji.branchIds.length > 0, 'actual annotated scene must yield branch instances');
    assert.ok(dji.branchIds.every(id => id <= dji.branchCount));
    assert.equal(dji.branchFrame, dji.frame);
    assert.ok(Number.isFinite(dji.branchCpuMs));
    assert.ok(dji.cameraUp[2] > 0.9, 'the DJI view should appear upright in its Z-up scene');
    assert.deepEqual(unexpectedBrowserErrors(), [], 'application should not report browser/render errors');

    const djiLinks = await evaluate(`(() => {
        const live=scene.events.invoke('annotation.panel').liveView,snapshot=scene.events.invoke('annotation.liveSnapshot');
        const mask=snapshot.mask.getContext('2d').getImageData(0,0,640,640).data;
        const left=snapshot.linkOverlay.getContext('2d').getImageData(0,0,640,640).data;
        const right=snapshot.depthLinkOverlay.getContext('2d').getImageData(0,0,640,640).data;
        let colored=0,outside=0,depthChanged=0,changedLabels=0;
        const colors=new Set(),ids=new Set(snapshot.linkLabels),depthIds=new Set(snapshot.detection.depthLinkLabels);
        for(let i=0;i<snapshot.linkLabels.length;i++)if(snapshot.linkLabels[i]!==snapshot.detection.depthLinkLabels[i])changedLabels++;
        for(let i=0;i<640*640;i++)if(left[i*4+3]){
            colored++;if(!mask[i*4])outside++;
            if(left[i*4]!==right[i*4]||left[i*4+1]!==right[i*4+1]||left[i*4+2]!==right[i*4+2])depthChanged++;
            colors.add([left[i*4],left[i*4+1],left[i*4+2]].join(','));
        }
        return {checked:live.junctionCheckbox.checked,count:snapshot.linkCount,
            depthCount:snapshot.detection.depthLinkCount,depthCuts:snapshot.detection.depthLinkCuts,
            armIds:snapshot.detection.depthJunctionArmIds,
            labelSize:snapshot.linkLabels.length,
            depthLabelSize:snapshot.detection.depthLinkLabels.length,depthUsed:depthIds.size-1,changedLabels,
            cropSize:snapshot.comparisonCrop?.size,
            used:ids.size-1,colored,outside,colors:colors.size,depthChanged,
            leftLabel:live.junctionNoDepthLabel.textContent,rightLabel:live.junctionDepthLabel.textContent};
    })()`);
    assert.equal(djiLinks.checked,false,'colored links should be the default');
    assert.equal(djiLinks.labelSize,320*320);
    assert.equal(djiLinks.depthLabelSize,320*320);
    assert.ok(djiLinks.cropSize<640,'the comparison should zoom into the raw skeleton');
    assert.ok(djiLinks.count>1 && djiLinks.used>1 && djiLinks.colors>1);
    assert.ok(djiLinks.depthCount>=djiLinks.count,
        'DA2 must never rejoin arms that were separated at a 2D junction');
    assert.ok(djiLinks.depthCuts>=0);
    for (const ids of djiLinks.armIds) {
        assert.ok(ids.every(Boolean),'every junction arm must have an object ID');
        assert.equal(new Set(ids).size,ids.length,'junction arms must have distinct object IDs');
    }
    assert.equal(djiLinks.depthUsed,djiLinks.depthCount);
    assert.ok(djiLinks.colored>0);
    if (djiLinks.depthCuts>0) assert.ok(djiLinks.changedLabels>0 && djiLinks.depthChanged>0);
    assert.equal(djiLinks.outside,0,'colored links must stay inside visible segmentation');
    assert.match(djiLinks.leftLabel,/colored links/);
    assert.match(djiLinks.rightLabel,/separate objects/);
    console.log(JSON.stringify({djiLinks:{count:djiLinks.count,depthCount:djiLinks.depthCount,
        depthCuts:djiLinks.depthCuts,changedLabels:djiLinks.changedLabels,
        armIds:djiLinks.armIds}}));
    const pca = await evaluate(`(() => {
        const live=scene.events.invoke('annotation.panel').liveView,s=scene.events.invoke('annotation.liveSnapshot');
        const input=s.detection.depthLinkLabels,merged=s.pcaMergedLabels;
        const before={frame:s.frameId,angle:live.pcaMaxAngle,input:s.detection.depthLinkCount,
            output:s.pcaMergedCount,objects:s.detection.pcaObjects.length,
            reliable:s.detection.pcaObjects.filter(o=>o.reliable).length,
            edges:s.detection.pcaAdjacency.length,eligible:s.detection.pcaAdjacency.filter(e=>e.eligible).length,
            inputLength:input.length,mergedLength:merged.length,
            axesCaption:live.pcaInputLabel.textContent,mergedCaption:live.pcaMergedLabel.textContent};
        const slider=live.pcaAngleSlider;
        slider.value='0';slider.dispatchEvent(new Event('input',{bubbles:true}));
        const atZero=s.pcaMergedCount;
        slider.value='90';slider.dispatchEvent(new Event('input',{bubbles:true}));
        const atNinety=s.pcaMergedCount;
        const changed=merged.some((id,i)=>id!==s.pcaMergedLabels[i]);
        const mask=s.mask.getContext('2d').getImageData(0,0,640,640).data;
        const overlay=s.pcaMergedOverlay.getContext('2d').getImageData(0,0,640,640).data;
        let outside=0,colored=0;
        for(let i=0;i<640*640;i++)if(overlay[i*4+3]){colored++;if(!mask[i*4])outside++;}
        slider.value='20';slider.dispatchEvent(new Event('input',{bubbles:true}));
        return {...before,atZero,atNinety,changed,colored,outside,
            sameFrame:s===scene.events.invoke('annotation.liveSnapshot'),
            numberValue:live.pcaAngleNumber.value};
    })()`);
    assert.equal(pca.angle,20);
    assert.equal(pca.inputLength,320*320);
    assert.equal(pca.mergedLength,320*320);
    assert.ok(pca.objects>0 && pca.reliable>0);
    assert.ok(pca.output<=pca.input);
    assert.ok(pca.atNinety<=pca.atZero);
    if (pca.eligible>0) assert.ok(pca.atNinety<pca.atZero && pca.changed);
    assert.ok(pca.colored>0 && pca.sameFrame);
    assert.equal(pca.outside,0);
    assert.equal(pca.numberValue,'20');
    assert.match(pca.axesCaption,/PCA axes/);
    assert.match(pca.mergedCaption,/PCA merged/);
    console.log(JSON.stringify({djiPca:pca}));
    const gravity = await evaluate(`(() => {
        const live=scene.events.invoke('annotation.panel').liveView,s=scene.events.invoke('annotation.liveSnapshot');
        const objects=s.gravityMergedObjects,up=s.pose.world_up_axis==='z'?2:1;
        const eligible=new Set(s.detection.pcaObjects.filter(o=>o.reliable &&
            90-Math.acos(Math.min(1,Math.abs(o.axis[up])))*180/Math.PI<=live.gravityTolerance).map(o=>o.id));
        let removedPixels=0,incorrectPixels=0;
        for(let i=0;i<s.detection.depthLinkLabels.length;i++){
            const source=s.detection.depthLinkLabels[i],kept=s.gravityMergedLabels[i];
            if(source&&!eligible.has(source)){removedPixels++;if(kept)incorrectPixels++;}
        }
        const angleError=Math.max(0,...objects.filter(o=>o.reliable).map(o=>Math.abs(
            o.angleToGravityDeg-Math.acos(Math.min(1,Math.abs(o.axis[up])))*180/Math.PI)));
        const before={frame:s.frameId,up:s.pose.world_up_axis,tolerance:live.gravityTolerance,
            merged:s.pcaMergedCount,objects:objects.length,reliable:objects.filter(o=>o.reliable).length,
            eligible:s.gravityEligibleCount,expectedEligible:eligible.size,removedPixels,incorrectPixels,
            selected:new Set(s.gravityFilteredLabels).size-1,
            minLength:live.minGravityLength,excludeTrunks:live.excludeGravityTrunks,
            trunkObjects:objects.filter(o=>o.isTrunk).length,
            lengths:objects.map(o=>o.centerlineLengthPx),
            angleError,inputLabel:live.gravityInputLabel.textContent,
            outputLabel:live.gravityFilteredLabel.textContent};
        const slider=live.gravityAngleSlider;
        slider.value='0';slider.dispatchEvent(new Event('input',{bubbles:true}));
        const atZero=new Set(s.gravityFilteredLabels).size-1;
        slider.value='90';slider.dispatchEvent(new Event('input',{bubbles:true}));
        const length=live.gravityLengthSlider;
        length.value='0';length.dispatchEvent(new Event('input',{bubbles:true}));
        const atNinety=new Set(s.gravityFilteredLabels).size-1;
        const atNinetyReliable=s.gravityMergedObjects.filter(o=>o.reliable).length;
        const atNinetyEligible=s.gravityEligibleCount;
        const wideObjects=s.gravityMergedObjects;
        length.value='200';length.dispatchEvent(new Event('input',{bubbles:true}));
        const atLong=new Set(s.gravityFilteredLabels).size-1;
        length.value='0';length.dispatchEvent(new Event('input',{bubbles:true}));
        const trunk=live.gravityTrunkCheckbox;
        trunk.checked=true;trunk.dispatchEvent(new Event('change',{bubbles:true}));
        const withoutTrunks=new Set(s.gravityFilteredLabels);withoutTrunks.delete(0);
        const expectedWithoutTrunks=wideObjects.filter(o=>o.reliable&&!o.isTrunk).map(o=>o.id);
        const mask=s.mask.getContext('2d').getImageData(0,0,640,640).data;
        const overlay=s.gravityFilteredOverlay.getContext('2d').getImageData(0,0,640,640).data;
        let colored=0,outside=0;
        for(let i=0;i<640*640;i++)if(overlay[i*4+3]){colored++;if(!mask[i*4])outside++;}
        trunk.checked=false;trunk.dispatchEvent(new Event('change',{bubbles:true}));
        length.value='12';length.dispatchEvent(new Event('input',{bubbles:true}));
        slider.value='20';slider.dispatchEvent(new Event('input',{bubbles:true}));
        const kept=new Set(s.gravityFilteredLabels);kept.delete(0);
        const expected=objects.filter(o=>o.reliable && 90-o.angleToGravityDeg<=20 &&
            o.centerlineLengthPx>=12).map(o=>o.id);
        return {...before,atZero,atNinety,atNinetyReliable,atNinetyEligible,atLong,colored,outside,
            withoutTrunkIds:[...withoutTrunks].sort((a,b)=>a-b),
            expectedWithoutTrunkIds:expectedWithoutTrunks.sort((a,b)=>a-b),
            selectedIds:[...kept].sort((a,b)=>a-b),expectedIds:expected.sort((a,b)=>a-b),
            sameFrame:s===scene.events.invoke('annotation.liveSnapshot'),
            sixthPanelUnchanged:s.pcaMergedCount===before.merged,
            sliderValue:live.gravityAngleSlider.value,numberValue:live.gravityAngleNumber.value,
            lengthValue:live.gravityLengthNumber.value,trunkChecked:live.gravityTrunkCheckbox.checked};
    })()`);
    assert.equal(gravity.tolerance,20);
    assert.equal(gravity.up,'z');
    assert.ok(gravity.objects<=gravity.eligible);
    assert.ok(gravity.reliable>0 && gravity.angleError<1e-8);
    assert.equal(gravity.eligible,gravity.expectedEligible);
    assert.ok(gravity.removedPixels>0);
    assert.equal(gravity.incorrectPixels,0,'excluded source components must not enter merged objects');
    assert.ok(gravity.atZero<=gravity.selected && gravity.selected<=gravity.atNinety);
    assert.equal(gravity.atNinety,gravity.atNinetyReliable);
    assert.equal(gravity.atNinetyEligible,pca.reliable);
    assert.ok(gravity.atLong<=gravity.atNinety);
    assert.ok(gravity.lengths.some(length=>length>0));
    assert.deepEqual(gravity.withoutTrunkIds,gravity.expectedWithoutTrunkIds);
    assert.deepEqual(gravity.selectedIds,gravity.expectedIds);
    assert.ok(gravity.colored>0 && gravity.sameFrame);
    assert.equal(gravity.outside,0);
    assert.equal(gravity.sliderValue,'20');
    assert.equal(gravity.numberValue,'20');
    assert.equal(gravity.lengthValue,'12');
    assert.equal(gravity.trunkChecked,false);
    assert.equal(gravity.minLength,12);
    assert.equal(gravity.excludeTrunks,false);
    assert.equal(gravity.sixthPanelUnchanged,true);
    assert.match(gravity.inputLabel,/Component gate/);
    assert.match(gravity.outputLabel,/Near-horizontal/);
    console.log(JSON.stringify({djiGravity:gravity}));
    await evaluate(`document.querySelector('.live-gravity-tile').scrollIntoView({block:'center'})`);
    const gravityShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_gravity_qa.png', Buffer.from(gravityShot.data, 'base64'));
    await evaluate(`document.querySelector('.live-pca-tile').scrollIntoView({block:'center'})`);
    const pcaShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_pca_qa.png', Buffer.from(pcaShot.data, 'base64'));
    await evaluate(`document.querySelector('.live-junction-tile').scrollIntoView({block:'center'})`);
    const linksShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_links_qa.png', Buffer.from(linksShot.data, 'base64'));
    await evaluate(`document.querySelector('[aria-label="Show junctions"]').click()`);

    const djiRaw = await evaluate(`(() => {
        const live=scene.events.invoke('annotation.panel').liveView,snapshot=scene.events.invoke('annotation.liveSnapshot');
        const raw=snapshot.rawSkeleton,mask=snapshot.mask.getContext('2d').getImageData(0,0,640,640).data;
        const display=live.branchCanvas.getContext('2d').getImageData(0,0,640,640).data;
        let marked=0,outside=0;
        for(let i=0;i<640*640;i++)if(display[i*4]===255&&display[i*4+1]===30&&display[i*4+2]===215){
            marked++;if(!mask[i*4])outside++;
        }
        return {mode:live.viewMode,width:raw?.length,count:snapshot.rawSkeletonCount,
            actual:raw?.reduce((sum,pixel)=>sum+pixel,0),marked,outside,
            junctions:snapshot.rawJunctions?.length,
            depthJunctions:snapshot.detection?.junctionComparison?.length,
            junctionStatuses:snapshot.detection?.junctionComparison?.reduce((counts,item)=>{
                counts[item.status]=(counts[item.status]||0)+1;return counts;
            },{}),
            junctionLabel:live.junctionNoDepthLabel.textContent,
            depthLabel:live.junctionDepthLabel.textContent,
            title:live.branchCanvas.parentElement.querySelector('.live-tile-title').textContent,
            topKVisible:getComputedStyle(document.querySelector('[aria-label="Landing candidate top K"]').parentElement).display,
            overlay:getComputedStyle(live.mainOverlay).visibility};
    })()`);
    assert.equal(djiRaw.mode,'raw');
    assert.equal(djiRaw.width,320*320);
    assert.equal(djiRaw.count,djiRaw.actual);
    assert.ok(djiRaw.marked>0,'raw skeleton must be visible over RGB');
    assert.equal(djiRaw.outside,0,'raw skeleton overlay must stay inside visible segmentation');
    assert.ok(djiRaw.junctions>0,'the tree skeleton should expose 2D junctions');
    assert.equal(djiRaw.depthJunctions,djiRaw.junctions,'depth comparison must classify the same raw junctions');
    assert.equal(Object.values(djiRaw.junctionStatuses).reduce((sum,count)=>sum+count,0),djiRaw.junctions);
    assert.match(djiRaw.junctionLabel,/2D only/);
    assert.match(djiRaw.depthLabel,/With DA2/);
    assert.match(djiRaw.title,/Raw skeleton/);
    assert.equal(djiRaw.topKVisible,'none');
    assert.equal(djiRaw.overlay,'hidden');
    console.log(JSON.stringify({djiJunctions:{count:djiRaw.junctions,statuses:djiRaw.junctionStatuses}}));
    await evaluate(`(() => {const tile=document.querySelector('.live-junction-tile');
        tile.scrollIntoView({block:'center'});return tile.getBoundingClientRect().top;})()`);
    const junctionShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_junction_qa.png', Buffer.from(junctionShot.data, 'base64'));
    await evaluate(`document.querySelector('[aria-label="Show junctions"]').click()`);
    assert.equal(await evaluate(`scene.events.invoke('annotation.panel').liveView.showJunctions`),false);
    await evaluate(`document.querySelector('.bbox-manager .bbox-preview-body').scrollTop=0`);
    await evaluate(`(() => {const select=document.querySelector('[aria-label="Branch visualization"]');
        select.value='candidates';select.dispatchEvent(new Event('change',{bubbles:true}));})()`);

    const expanded = await evaluate(`(() => {
        const panel=document.querySelector('.bbox-manager');
        panel.style.width='620px';panel.style.height='800px';panel.classList.add('resized');
        const before=panel.getBoundingClientRect();
        const topK=document.querySelector('[aria-label="Landing candidate top K"]');
        const hiddenBefore=getComputedStyle(topK.parentElement).display;
        panel.querySelector('[aria-label="Maximize BBox Manager"]').click();
        const full=panel.getBoundingClientRect();
        const visibleMaximized=getComputedStyle(topK.parentElement).display;
        const live=scene.events.invoke('annotation.panel').liveView;
        const pcaControl=live.pcaAngleSlider.getBoundingClientRect();
        const gravityControl=live.gravityAngleSlider.getBoundingClientRect();
        const gravityLengthControl=live.gravityLengthSlider.getBoundingClientRect();
        const gravityTrunkControl=live.gravityTrunkCheckbox.getBoundingClientRect();
        live.pcaAngleSlider.value='35';live.pcaAngleSlider.dispatchEvent(new Event('input',{bubbles:true}));
        const pcaMaximized={visible:pcaControl.width>0,angle:live.pcaMaxAngle,
            merged:scene.events.invoke('annotation.liveSnapshot').pcaMergedCount,
            gravityVisible:gravityControl.width>0,
            gravityLengthVisible:gravityLengthControl.width>0,
            gravityTrunkVisible:gravityTrunkControl.width>0,
            gravityObjects:scene.events.invoke('annotation.liveSnapshot').pcaMergedObjects.length};
        live.gravityAngleSlider.value='30';live.gravityAngleSlider.dispatchEvent(new Event('input',{bubbles:true}));
        live.gravityLengthSlider.value='30';live.gravityLengthSlider.dispatchEvent(new Event('input',{bubbles:true}));
        live.gravityTrunkCheckbox.checked=true;live.gravityTrunkCheckbox.dispatchEvent(new Event('change',{bubbles:true}));
        const gravityMaximized={tolerance:live.gravityTolerance,
            minLength:live.minGravityLength,excludeTrunks:live.excludeGravityTrunks,
            selected:new Set(scene.events.invoke('annotation.liveSnapshot').gravityFilteredLabels).size-1};
        live.gravityTrunkCheckbox.checked=false;live.gravityTrunkCheckbox.dispatchEvent(new Event('change',{bubbles:true}));
        live.gravityLengthSlider.value='12';live.gravityLengthSlider.dispatchEvent(new Event('input',{bubbles:true}));
        live.gravityAngleSlider.value='20';live.gravityAngleSlider.dispatchEvent(new Event('input',{bubbles:true}));
        live.pcaAngleSlider.value='20';live.pcaAngleSlider.dispatchEvent(new Event('input',{bubbles:true}));
        const camera=document.getElementById('sensor-frame').getBoundingClientRect();
        return {before:[before.width,before.height],full:[full.width,full.height],panelLeft:full.left,camera:[camera.left,camera.right],
            hiddenBefore,visibleMaximized,pcaMaximized,gravityMaximized,
            split:document.body.classList.contains('bbox-manager-camera-split')};
    })()`);
    assert.ok(expanded.full[0] > expanded.before[0] && expanded.full[1] > expanded.before[1]);
    assert.ok(expanded.camera[1] < expanded.panelLeft, 'camera stays visible beside the expanded panel');
    assert.equal(expanded.split, true);
    assert.notEqual(expanded.hiddenBefore,'none','Top K must be available at the normal panel size');
    assert.notEqual(expanded.visibleMaximized,'none');
    assert.equal(expanded.pcaMaximized.visible,true);
    assert.equal(expanded.pcaMaximized.angle,35);
    assert.ok(expanded.pcaMaximized.merged>0);
    assert.equal(expanded.pcaMaximized.gravityVisible,true);
    assert.equal(expanded.pcaMaximized.gravityLengthVisible,true);
    assert.equal(expanded.pcaMaximized.gravityTrunkVisible,true);
    assert.equal(expanded.pcaMaximized.gravityObjects,expanded.pcaMaximized.merged);
    assert.equal(expanded.gravityMaximized.tolerance,30);
    assert.equal(expanded.gravityMaximized.minLength,30);
    assert.equal(expanded.gravityMaximized.excludeTrunks,true);
    await delay(300);
    const maxView = await evaluate(`(() => {
        const camera=scene.camera,rect=document.getElementById('canvas-container').getBoundingClientRect();
        const x=Math.round(rect.left+rect.width*.78),y=Math.round(rect.top+rect.height*.5);
        document.querySelector('[aria-label="Restore BBox Manager"]').focus();
        window._maxFlyEvents=[];
        scene.events.on('camera.fly.forward',down=>window._maxFlyEvents.push(down));
        const p=camera.mainCamera.getPosition();
        const f=camera.focalPoint;
        return {x,y,hit:!!document.elementFromPoint(x,y)?.closest('#canvas-container'),azim:camera.azim,position:[p.x,p.y,p.z],focal:[f.x,f.y,f.z],control:camera.controlMode};
    })()`);
    assert.equal(maxView.hit,true,'the main camera must receive pointer input while the manager is maximized');
    await cdp.send('Input.dispatchKeyEvent',{type:'keyDown',key:'w',code:'KeyW',windowsVirtualKeyCode:87});
    // The large DJI splat renders very slowly in headless SwiftShader. Advance one
    // camera update while W is held to verify the actual shortcut/controller path.
    const keyMoved=await evaluate(`(() => {scene.camera.onUpdate(.2);const p=scene.camera.mainCamera.getPosition();return [p.x,p.y,p.z];})()`);
    await cdp.send('Input.dispatchKeyEvent',{type:'keyUp',key:'w',code:'KeyW',windowsVirtualKeyCode:87});
    assert.deepEqual(await evaluate('window._maxFlyEvents'),[true,false]);
    await cdp.send('Input.dispatchMouseEvent',{type:'mousePressed',x:maxView.x,y:maxView.y,button:'left',buttons:1,clickCount:1});
    await cdp.send('Input.dispatchMouseEvent',{type:'mouseMoved',x:maxView.x+55,y:maxView.y,button:'left',buttons:1});
    await cdp.send('Input.dispatchMouseEvent',{type:'mouseReleased',x:maxView.x+55,y:maxView.y,button:'left',buttons:0,clickCount:1});
    await delay(150);
    const maxMoved = await evaluate(`(() => {
        const camera=scene.camera,p=camera.mainCamera.getPosition();
        const panel=document.querySelector('.bbox-manager');
        panel.querySelector('[aria-label="Restore BBox Manager"]').click();
        const restored=panel.getBoundingClientRect();
        return {azim:camera.azim,position:[p.x,p.y,p.z],restored:[restored.width,restored.height],
            split:document.body.classList.contains('bbox-manager-camera-split')};
    })()`);
    assert.ok(Math.hypot(...keyMoved.map((value,index)=>value-maxView.position[index]))>.001,'W should move the camera with the manager maximized');
    assert.ok(Math.abs(maxMoved.azim-maxView.azim)>1,'mouse drag should turn the camera with the manager maximized');
    assert.ok(Math.abs(maxMoved.restored[0]-expanded.before[0])<2 && Math.abs(maxMoved.restored[1]-expanded.before[1])<2);
    assert.equal(maxMoved.split,false);
    await evaluate(`(() => {
        const camera=scene.camera,V=camera.mainCamera.getPosition().constructor;
        camera.setPose(new V(...${JSON.stringify(maxView.position)}),new V(...${JSON.stringify(maxView.focal)}),0);
        camera.onUpdate(0);
        scene.events.invoke('annotation.panel').liveView.refresh();
    })()`);
    await waitFor('document.querySelector(".live-view")?.dataset.state === "ready" && scene.events.invoke("annotation.liveSnapshot")?.frameId !== '+JSON.stringify(dji.frame),90000);
    const yellow = await evaluate(`(() => {
        const panel=scene.events.invoke('annotation.panel'),live=panel.liveView,snapshot=scene.events.invoke('annotation.liveSnapshot');
        const data=snapshot.mask.getContext('2d').getImageData(0,0,640,640).data;
        let index=0;while(index<640*640 && !data[index*4])index++;
        const x=index%640,y=Math.floor(index/640),ctx=live.maskCanvas.getContext('2d');
        const before=[...ctx.getImageData(x,y,1,1).data];
        const toggle=document.querySelector('[aria-label="Show yellow segmentation"]');
        toggle.checked=false;toggle.dispatchEvent(new Event('change',{bubbles:true}));
        const hidden=[...ctx.getImageData(x,y,1,1).data];
        const eventOff=scene.events.invoke('annotation.showSegmentationTint');
        const titleOff=live.maskCanvas.parentElement.querySelector('.live-tile-title').textContent;
        toggle.checked=true;toggle.dispatchEvent(new Event('change',{bubbles:true}));
        return {before,hidden,restored:[...ctx.getImageData(x,y,1,1).data],
            eventOff,eventOn:scene.events.invoke('annotation.showSegmentationTint'),titleOff,
            maskValue:data[index*4],frame:snapshot.frameId};
    })()`);
    assert.equal(yellow.maskValue,255);
    assert.notDeepEqual(yellow.hidden,yellow.before,'yellow overlay toggle must change the segmentation preview');
    assert.deepEqual(yellow.restored,yellow.before);
    assert.equal(yellow.eventOff,false);
    assert.equal(yellow.eventOn,true);
    assert.match(yellow.titleOff,/hidden/);
    assert.equal(yellow.frame,await evaluate('scene.events.invoke("annotation.liveSnapshot").frameId'));
    const filtered = await evaluate(`(() => {
        const panel=scene.events.invoke('annotation.panel'),live=panel.liveView,snapshot=scene.events.invoke('annotation.liveSnapshot');
        window._qaOriginalFetch=window.fetch.bind(window);window._qaDepthCalls=0;
        window.fetch=(...args)=>{if(args[0]==='/api/depth')window._qaDepthCalls++;return window._qaOriginalFetch(...args);};
        const length=document.querySelector('[aria-label="Minimum branch length"]');
        length.value='80';length.dispatchEvent(new Event('input',{bubbles:true}));
        const strictCount=live.branchView(snapshot).horizontal.length;
        [...document.querySelectorAll('.landing-search button')].find(button=>button.textContent==='Reset search').click();
        const shape=document.querySelector('[aria-label="Minimum branch length to width ratio"]');
        shape.value='5';shape.dispatchEvent(new Event('input',{bubbles:true}));
        const wideBlobCount=live.branchView(snapshot).horizontal.length;
        [...document.querySelectorAll('.landing-search button')].find(button=>button.textContent==='Reset search').click();
        const certainty=document.querySelector('[aria-label="Minimum direction confidence"]');
        certainty.value='1';certainty.dispatchEvent(new Event('input',{bubbles:true}));
        const highCertaintyCount=live.branchView(snapshot).horizontal.length;
        [...document.querySelectorAll('.landing-search button')].find(button=>button.textContent==='Reset search').click();
        const resetSettings=[live.horizontalAngle,live.minBranchLength,live.minLengthWidthRatio,live.minDirectionConfidence,live.excludeTrunk];
        const trunk=document.querySelector('[aria-label="Exclude trunk"]');trunk.checked=true;trunk.dispatchEvent(new Event('change',{bubbles:true}));
        const angle=document.querySelector('[aria-label="Maximum ground-parallel angle"]');angle.value='45';angle.dispatchEvent(new Event('input',{bubbles:true}));
        const topK=document.querySelector('[aria-label="Landing candidate top K"]');topK.value='1';topK.dispatchEvent(new Event('input',{bubbles:true}));
        const view=live.branchView(snapshot),ids=[...new Set(view.instanceMask)];
        const overlay=document.querySelector('#landing-overlay');
        const overlayRect=overlay.getBoundingClientRect(),cameraRect=document.querySelector('#canvas-container').getBoundingClientRect();
        const pixels=overlay.getContext('2d').getImageData(0,0,overlay.width,overlay.height).data;
        let markedPixels=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])markedPixels++;
        return {frame:snapshot.frameId,rawCount:snapshot.detection.branches.length,excluded:view.excludedTrunkIds,
            visibleCount:view.branches.length,horizontal:view.horizontal.map(b=>b.id),landing:view.landing.map(b=>b.id),
            landingMaskIds:[...new Set(view.landingMask)].filter(Boolean),ids,
            settings:[live.excludeTrunk,live.horizontalAngle,live.landingTopK],strictCount,wideBlobCount,highCertaintyCount,resetSettings,
            list:live.branchList.textContent,
            overlayVisible:getComputedStyle(overlay).visibility,overlayFrame:overlay.dataset.frameId,markedPixels,
            overlayRect:[overlayRect.x,overlayRect.y,overlayRect.width,overlayRect.height],
            cameraRect:[cameraRect.x,cameraRect.y,cameraRect.width,cameraRect.height]};
    })()`);
    await delay(700);
    assert.deepEqual(filtered.settings,[true,45,1]);
    assert.equal(filtered.strictCount,0,'length slider must remove short candidates immediately');
    assert.equal(filtered.wideBlobCount,0,'shape slider must remove broad candidates immediately');
    assert.equal(filtered.highCertaintyCount,0,'certainty slider must remove uncertain candidates immediately');
    assert.deepEqual(filtered.resetSettings,[25,12,1.5,0.5,false]);
    assert.ok(filtered.excluded.length > 0, 'actual tree must yield trunk IDs to exclude');
    assert.ok(filtered.horizontal.length > 0, 'actual tree must retain a ground-parallel branch candidate');
    assert.deepEqual(filtered.landingMaskIds,filtered.landing,'the displayed landing mask must only contain selected IDs');
    assert.equal(filtered.overlayVisible,'visible');
    assert.equal(filtered.overlayFrame,filtered.frame);
    assert.ok(filtered.markedPixels > 0);
    filtered.overlayRect.forEach((value,index)=>assert.ok(Math.abs(value-filtered.cameraRect[index])<2,'main overlay must align with the camera viewport'));
    assert.equal(filtered.visibleCount, filtered.rawCount-filtered.excluded.length);
    assert.ok(filtered.excluded.every(id => !filtered.ids.includes(id)), 'excluded trunks must be zeroed in the displayed ID map');
    assert.equal(await evaluate('window._qaDepthCalls'),0,'filter controls must not re-run DA2');
    assert.equal(await evaluate('scene.events.invoke("annotation.liveSnapshot").frameId'),filtered.frame);
    const mainToggle = await evaluate(`(() => {
        const input=document.querySelector('[aria-label="Show landing candidates on main view"]');
        input.checked=false;input.dispatchEvent(new Event('change',{bubbles:true}));
        const off=getComputedStyle(document.querySelector('#landing-overlay')).visibility;
        input.checked=true;input.dispatchEvent(new Event('change',{bubbles:true}));
        return {off,on:getComputedStyle(document.querySelector('#landing-overlay')).visibility};
    })()`);
    assert.deepEqual(mainToggle,{off:'hidden',on:'visible'});
    await evaluate('window.fetch=window._qaOriginalFetch');
    const filteredShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_filtered_qa.png', Buffer.from(filteredShot.data, 'base64'));
    console.log(JSON.stringify({ expanded, filtered }));

    await evaluate(`(() => {
        window._qaFiles=[];
        const directory=(name)=>({name,getDirectoryHandle:async(child)=>directory(child),getFileHandle:async(file)=>({createWritable:async()=>({write:async(blob)=>{
            const bytes=new Uint8Array(await blob.arrayBuffer()); let text='';
            for(let start=0;start<bytes.length;start+=8192) text+=String.fromCharCode(...bytes.subarray(start,start+8192));
            window._qaFiles.push({path:name+'/'+file,base64:btoa(text)});
        },close:async()=>{}})})});
        scene.events.invoke('annotation.panel').outputDirHandle=directory('root');
        return true;
    })()`);
    await evaluate(`window.scene.events.invoke('annotation.panel').liveView.save()`);
    await waitFor('window._qaFiles.length === 7');
    const djiFiles = await evaluate('window._qaFiles');
    const detectionFile = djiFiles.find(file => file.path.startsWith('Branches/') && file.path.endsWith('.json'));
    const savedDetection = JSON.parse(Buffer.from(detectionFile.base64,'base64').toString('utf8'));
    assert.equal(savedDetection.filters.exclude_trunk,true);
    assert.equal(savedDetection.filters.horizontal_angle_max_deg,45);
    assert.equal(savedDetection.filters.min_branch_length_px,12);
    assert.equal(savedDetection.filters.min_length_width_ratio,1.5);
    assert.equal(savedDetection.filters.min_direction_confidence,0.5);
    assert.equal(savedDetection.filters.landing_top_k,1);
    assert.deepEqual(savedDetection.excluded_trunk_ids,filtered.excluded);
    assert.deepEqual(savedDetection.horizontal_branch_ids,filtered.horizontal);
    assert.deepEqual(savedDetection.landing_candidate_ids,filtered.landing);
    for (const file of djiFiles) {
        const filename = path.join('/tmp/annotation_live_dji_export_qa', file.path);
        await mkdir(path.dirname(filename), { recursive: true });
        await writeFile(filename, Buffer.from(file.base64, 'base64'));
    }
    console.log(JSON.stringify({ djiExportFiles: djiFiles.map(file => file.path) }));

    await evaluate(`(() => {
        const manager=document.querySelector('.bbox-manager');
        [...manager.querySelectorAll('.bbox-preview-head button')].find(button=>button.textContent==='hide').click();
        const position=scene.camera.mainCamera.getPosition().clone();position.x+=.04;
        const matrix=scene.events.invoke('annotation.liveSnapshot').pose.camera_to_world;
        const target=position.clone().set(position.x-matrix[0][2],position.y-matrix[1][2],position.z-matrix[2][2]);
        scene.events.fire('camera.setPose',{position,target},0);
        scene.forceRender=true;
    })()`);
    await waitFor(`getComputedStyle(document.querySelector('#landing-overlay')).visibility === 'hidden'`, 10000);
    assert.equal(await evaluate(`document.querySelector('#landing-overlay').dataset.frameId`),'','moving the camera must clear the old overlay frame');
    await waitFor(`scene.events.invoke('annotation.liveSnapshot')?.frameId !== ${JSON.stringify(filtered.frame)} && scene.events.invoke('annotation.liveSnapshot')?.detection`, 240000);
    assert.equal(await evaluate(`document.querySelector('.bbox-manager').classList.contains('collapsed')`),true);
    assert.equal(await evaluate(`getComputedStyle(document.querySelector('.bbox-manager .bbox-preview-body')).display`),'none');
    assert.ok(await evaluate(`document.querySelector('.bbox-manager').getBoundingClientRect().height < 100`));
    const hiddenPanelLanding = await evaluate(`(() => {
        const panel=scene.events.invoke('annotation.panel'),snapshot=scene.events.invoke('annotation.liveSnapshot');
        return {frame:snapshot.frameId,landing:panel.liveView.branchView(snapshot).landing.length,
            overlay:getComputedStyle(document.querySelector('#landing-overlay')).visibility};
    })()`);
    assert.equal(hiddenPanelLanding.overlay,hiddenPanelLanding.landing ? 'visible' : 'hidden');
    const mainShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_dji_main_overlay_qa.png', Buffer.from(mainShot.data, 'base64'));
    console.log(JSON.stringify({ hiddenPanelLanding }));

    // Restore the previous manually authored annotation through the shareable URL.
    await cdp.send('Page.navigate', { url: `${djiUrl}&boxes3d=/annotations/dji_0967_saved_boxes_3d.json` });
    await waitFor('window.scene?.events?.invoke("annotation.panel")?.boxes.length === 8 && window.scene.events.invoke("annotation.liveSnapshot")?.pose.segmentation.annotations.length === 8', 240000);
    const restored = await evaluate(`(() => {
        const panel=scene.events.invoke('annotation.panel');
        const snapshot=scene.events.invoke('annotation.liveSnapshot');
        const pixels=snapshot.mask.getContext('2d').getImageData(0,0,640,640).data;
        let maskArea=0;
        for(let i=0;i<pixels.length;i+=4) if(pixels[i]) maskArea++;
        return {ids:panel.boxes.map(box=>box.id),firstCenter:panel.boxes[0].center,metricScale:panel.metersPerSceneUnit,pinned:panel.metricScalePinnedByUrl,maskArea};
    })()`);
    const restoredShot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    await writeFile('/tmp/annotation_live_restored_qa.png', Buffer.from(restoredShot.data, 'base64'));
    console.log(JSON.stringify({ restored }));
    assert.equal(restored.ids[0], 'branch_001');
    assert.deepEqual(restored.firstCenter, [-0.1134, 0.3784, 0.0263]);
    assert.ok(Math.abs(restored.metricScale - 1.5118769969045391) < 1e-6);
    assert.equal(restored.pinned, true);
    assert.deepEqual(unexpectedBrowserErrors(), [], 'restored annotations should not report browser/render errors');
} finally {
    cdp?.ws.close();
    chrome.kill('SIGTERM');
    await new Promise(resolve => chrome.once('exit', resolve));
    await rm(profile, { recursive: true, force: true });
    await rm(fixtureDir, { recursive: true, force: true });
}
