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

    await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: `
        window._qaDepthMode=location.href.includes('visibility.ply')?'step':'real';
        window._qaDepthCalls=0;
        const original=window.fetch.bind(window);
        window.fetch=async (...args)=>{
            if(args[0]!=='/api/depth') return original(...args);
            window._qaDepthCalls++;
            if(window._qaDepthMode==='real') return original(...args);
            if(window._qaDepthMode==='error') throw new Error('QA DA2 unavailable');
            if(window._qaDepthMode==='hold') await new Promise(resolve=>window._qaRelease=resolve);
            const depth=new Float32Array(640*640);
            for(let i=0;i<depth.length;i++)depth[i]=i%640<320?1:10;
            let raw='';const bytes=new Uint8Array(depth.buffer);
            for(let i=0;i<bytes.length;i+=8192)raw+=String.fromCharCode(...bytes.subarray(i,i+8192));
            return new Response(JSON.stringify({frame_id:JSON.parse(args[1].body).frame_id,
                width:640,height:640,dtype:'<f4',depth_f32_base64:btoa(raw),model:'qa-step',
                depth_kind:'raw_da2_relative_output',inference_ms:0,input_size:518}),{status:200,headers:{'Content-Type':'application/json'}});
        };` });
    const syntheticUrl=`http://127.0.0.1:8027/3dgs-splat-labeler/app/?load=../tests/${path.basename(fixtureDir)}/visibility.ply&filename=visibility.ply`;
    await cdp.send('Page.navigate',{url:syntheticUrl});
    await waitFor('window.scene?.events?.functions.has("annotation.panel") && scene.getElementsByType("splat").length');
    await evaluate(`(()=>{
        const p=scene.events.invoke('annotation.panel');
        p.boxes=[{id:'comparison_fixture',label:'branch',operation:'include',center:[0,0,.4],size:[2,2,2],euler:[0,0,0]}];
        p.box=p.boxes[0];p.activeBoxIndex=0;p.touched=true;p.writeInputs();
        const position=scene.camera.mainCamera.getPosition().clone().set(0,0,3),target=position.clone().set(0,0,0);
        scene.events.fire('camera.setPose',{position,target},0);scene.forceRender=true;return true;
    })()`);
    const ready=`(()=>{const s=scene.events.invoke('annotation.liveSnapshot');return s?.analyses?.da2?.analysisStatus==='ready' && s?.analyses?.['3dgs']?.analysisStatus==='ready'})()`;
    await waitFor(ready);
    const synthetic=await evaluate(`(()=>{
        const s=scene.events.invoke('annotation.liveSnapshot'),a=s.analyses.da2,b=s.analyses['3dgs'];
        let different=0,outside=0;const original=s.mask.getContext('2d').getImageData(0,0,640,640).data;
        for(let i=0;i<a.rawSkeleton.length;i++)if(a.rawSkeleton[i]!==b.rawSkeleton[i])different++;
        for(const v of [a,b]){const mask=v.mask.getContext('2d').getImageData(0,0,640,640).data;
            for(let i=0;i<640*640;i++)if((mask[i*4]||v.finalLanding.ids[i])&&!original[i*4])outside++;}
        return {frame:s.frameId,sourceFrames:[a.frameId,b.frameId],different,outside,
            skeletons:[a.rawSkeletonCount,b.rawSkeletonCount],boundaries:[a.boundaryCells,b.boundaryCells],
            units:[a.detection.geometry_units,b.detection.geometry_units],
            canvases:document.querySelectorAll('.depth-comparison canvas[data-stage]').length,
            overlay:document.querySelector('#landing-overlay').dataset.depthSource};
    })()`);
    assert.equal(synthetic.canvases,14);
    assert.ok(synthetic.different>0,'controlled DA2 step must change the skeleton before branch filtering');
    assert.equal(synthetic.outside,0);
    assert.deepEqual(synthetic.sourceFrames,[synthetic.frame,synthetic.frame]);
    assert.deepEqual(synthetic.units,['relative_pseudo3d','approximate_meters']);
    assert.equal(synthetic.overlay,'3dgs');
    console.log(JSON.stringify({synthetic}));

    // A declared box side drives depth and arc length, and survives annotation export/reload.
    const beforeCalibration=await evaluate(`(()=>{const p=scene.events.invoke('annotation.panel'),s=p.liveView.last;
        const index=s.gsDepth.findIndex(Number.isFinite);
        const before={frame:s.frameId,scale:p.metersPerSceneUnit,index,depth:s.gsDepth[index],length:s.analyses['3dgs'].skeletonLengthCm};
        p.boxCalibrationAxis.value='0';p.boxCalibrationMetersInput.value=String(p.box.size[0]*p.metersPerSceneUnit*2);
        p.boxCalibrationMetersInput.dispatchEvent(new Event('change'));return before;})()`);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').frameId!==${JSON.stringify(beforeCalibration.frame)} && scene.events.invoke('annotation.liveSnapshot').pose.gs_depth.scale_reference?.box_id==='comparison_fixture'`);
    const calibrated=await evaluate(`(()=>{const p=scene.events.invoke('annotation.panel'),s=p.liveView.last;
        return {scale:p.metersPerSceneUnit,depth:s.gsDepth[${beforeCalibration.index}],length:s.analyses['3dgs'].skeletonLengthCm,
            reference:s.pose.gs_depth.scale_reference,payload:p.payload(),url:location.href,
            caption:p.liveView.comparisonView.tiles.get('3dgs:skeleton').caption.textContent};})()`);
    assert.ok(beforeCalibration.length>0);
    assert.ok(Math.abs(calibrated.scale/beforeCalibration.scale-2)<1e-6);
    assert.ok(Math.abs(calibrated.depth/beforeCalibration.depth-2)<.01);
    assert.ok(Math.abs(calibrated.length/beforeCalibration.length-2)<.01);
    assert.deepEqual(calibrated.payload.metric_scale.reference,calibrated.reference);
    assert.match(calibrated.caption,/cm visible skeleton/);
    await writeFile(path.join(fixtureDir,'measured_boxes.json'),JSON.stringify(calibrated.payload));
    await cdp.send('Page.navigate',{url:calibrated.url+`&boxes3d=../tests/${path.basename(fixtureDir)}/measured_boxes.json`});
    await waitFor('window.scene?.events?.functions.has("annotation.panel") && scene.getElementsByType("splat").length');
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').pose.gs_depth.scale_reference?.box_id==='comparison_fixture' && scene.events.invoke('annotation.liveSnapshot').pose.segmentation.annotations[0]?.id==='comparison_fixture'`);
    const reloaded=await evaluate(`(()=>{const p=scene.events.invoke('annotation.panel');return {scale:p.metersPerSceneUnit,
        reference:p.boxScaleReference,depthReference:p.liveView.last.pose.gs_depth.scale_reference};})()`);
    assert.equal(reloaded.scale,calibrated.scale);assert.deepEqual(reloaded.reference,calibrated.reference);
    assert.deepEqual(reloaded.depthReference,calibrated.reference);
    console.log(JSON.stringify({boxCalibration:{depthRatio:calibrated.depth/beforeCalibration.depth,
        lengthRatio:calibrated.length/beforeCalibration.length,reference:calibrated.reference,reloadVerified:true}}));

    // Shared controls use cached depth and isolate caches by source and revision.
    const controls=await evaluate(`(()=>{
        const p=scene.events.invoke('annotation.panel'),l=p.liveView;
        window._callsBefore=window._qaDepthCalls;window._frameBefore=l.last.frameId;
        l.pcaAngleNumber.value='35';l.pcaAngleNumber.dispatchEvent(new Event('change'));
        const gap=document.querySelector('[aria-label="Shared skeleton depth gap"]');
        gap.value='.12';gap.dispatchEvent(new Event('change'));gap.value='.2';gap.dispatchEvent(new Event('change'));
        return {revision:l.analysisRevision};
    })()`);
    await waitFor(ready);
    const shared=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView,s=l.last;return {
        calls:window._qaDepthCalls-window._callsBefore,sameFrame:s.frameId===window._frameBefore,
        revisions:[s.analyses.da2.analysisRevision,s.analyses['3dgs'].analysisRevision],
        angles:[s.analyses.da2.pcaMergedAngle,s.analyses['3dgs'].pcaMergedAngle]};})()`);
    assert.equal(shared.calls,0);assert.equal(shared.sameFrame,true);
    assert.deepEqual(shared.revisions,[controls.revision,controls.revision]);
    assert.deepEqual(shared.angles,[35,35]);
    console.log(JSON.stringify({shared}));

    // A failed DA2 request must leave the 3DGS result visible and usable.
    await evaluate(`window._qaDepthMode='error';scene.events.invoke('annotation.panel').liveView.refresh()`);
    await waitFor(`(()=>{const s=scene.events.invoke('annotation.liveSnapshot');return s?.analyses?.da2?.analysisStatus==='error'&&s?.analyses?.['3dgs']?.analysisStatus==='ready'})()`);
    assert.equal(await evaluate(`getComputedStyle(document.querySelector('#landing-overlay')).visibility`),'visible');
    assert.equal(await evaluate(`document.querySelector('.live-save button').disabled`),true);
    assert.equal(await evaluate(`scene.events.invoke('annotation.panel').liveView.figuresButton.disabled`),false,
        'figures remain exportable when DA2 fails and 3DGS is ready');
    await evaluate(`window._qaDepthMode='step';scene.events.invoke('annotation.panel').liveView.refresh()`);
    await waitFor(ready);

    // Hold an old response, move the camera and verify both new source poses match.
    await evaluate(`window._qaDepthMode='hold';window._qaRelease=null;scene.events.invoke('annotation.panel').liveView.refresh()`);
    await waitFor('window._qaRelease');
    const oldFrame=await evaluate(`scene.events.invoke('annotation.liveSnapshot').frameId`);
    await evaluate(`(()=>{const position=scene.camera.mainCamera.getPosition().clone();position.x+=.08;
        const target=position.clone().set(0,0,0);scene.events.fire('camera.setPose',{position,target},0);
        window._qaDepthMode='step';window._qaRelease();scene.forceRender=true;return true;})()`);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').frameId!==${JSON.stringify(oldFrame)}`);
    const moved=await evaluate(`(()=>{const s=scene.events.invoke('annotation.liveSnapshot');return {
        frame:s.frameId,a:s.analyses.da2.pose.position,b:s.analyses['3dgs'].pose.position,
        overlay:document.querySelector('#landing-overlay').dataset.frameId};})()`);
    assert.deepEqual(moved.a,moved.b);assert.equal(moved.overlay,moved.frame);
    const shot=await cdp.send('Page.captureScreenshot',{format:'png'});
    await writeFile('/tmp/depth_comparison_synthetic.png',Buffer.from(shot.data,'base64'));

    // Run the actual DJI PLY, renderer depth and DA2 service.
    const djiUrl='http://127.0.0.1:8027/3dgs-splat-labeler/app/?load=../../exports/dji_0967_supersplat/dji_0967_splat.ply&filename=dji_0967_splat.ply&metricScale=../../data/dji_0967_3dgs/metric_scale.json&control=fly';
    await cdp.send('Page.navigate',{url:djiUrl});
    await waitFor('window.scene?.events?.functions.has("annotation.panel") && scene.getElementsByType("splat").length',240000);
    await waitFor(ready,240000);
    await evaluate(`(async()=>{const p=scene.events.invoke('annotation.panel'),picked=await scene.camera.intersect(.5,.72);
        if(!picked)throw new Error('No DJI surface');const v=picked.position;
        p.boxes=[{id:'dji_compare',label:'branch',operation:'include',center:[v.x,v.y,v.z],size:[.6,.6,.6],euler:[0,0,0]}];
        p.box=p.boxes[0];p.activeBoxIndex=0;p.touched=true;p.writeInputs();scene.forceRender=true;return true;})()`);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').pose.segmentation.annotations[0]?.id==='dji_compare'`,240000);
    const dji=await evaluate(`(()=>{const s=scene.events.invoke('annotation.liveSnapshot'),l=scene.events.invoke('annotation.panel').liveView;
        return {frame:s.frameId,summary:l.comparisonView.summary.textContent,
            sources:Object.fromEntries(['da2','3dgs'].map(k=>{const a=s.analyses[k];return [k,{
                frame:a.frameId,skeleton:a.rawSkeletonCount,regions:a.detection.depthLinkCount,
                final:a.finalLanding.selected.map(x=>x.id),eligible:a.finalLanding.eligible.length,
                valid:a.validFraction,boundaries:a.boundaryCells,units:a.detection.geometry_units}]})),
            tiles:[...document.querySelectorAll('.compare-skeleton canvas')].map(x=>{const r=x.getBoundingClientRect();return [r.left,r.width]})};})()`);
    assert.equal(dji.sources.da2.frame,dji.sources['3dgs'].frame);
    assert.ok(dji.sources.da2.skeleton>0&&dji.sources['3dgs'].skeleton>0);
    assert.ok(dji.tiles[1][0]>dji.tiles[0][0]);
    assert.match(dji.summary,/IoU|both masks empty/);
    console.log(JSON.stringify({dji}));

    const cmFilterBefore=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView,s=l.last;
        const before={frame:s.frameId,calls:window._qaDepthCalls,min:l.minSkeletonLengthCm,
            daIds:s.analyses.da2.finalLanding.selected.map(x=>x.id),length:s.analyses['3dgs'].skeletonLengthCm};
        l.gsLengthNumber.value='1000000';l.gsLengthNumber.dispatchEvent(new Event('change'));return before;})()`);
    await waitFor(`scene.events.invoke('annotation.liveSnapshot').analyses['3dgs'].finalLanding.selected.length===0`);
    const cmFilterAfter=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView,s=l.last;
        return {frame:s.frameId,calls:window._qaDepthCalls,daIds:s.analyses.da2.finalLanding.selected.map(x=>x.id),
            gsRegions:[...new Set(s.analyses['3dgs'].gravityFilteredLabels)].filter(Boolean).length,
            cmCaption:l.comparisonView.tiles.get('3dgs:gravity').caption.textContent};})()`);
    assert.ok(cmFilterBefore.length>0);
    assert.equal(cmFilterAfter.frame,cmFilterBefore.frame);assert.equal(cmFilterAfter.calls,cmFilterBefore.calls);
    assert.deepEqual(cmFilterAfter.daIds,cmFilterBefore.daIds);assert.equal(cmFilterAfter.gsRegions,0);
    assert.match(cmFilterAfter.cmCaption,/1000000 cm/);
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;l.gsLengthNumber.value=${JSON.stringify(String(10))};l.gsLengthNumber.dispatchEvent(new Event('change'));return true;})()`);
    await waitFor(`scene.events.invoke('annotation.panel').liveView.comparisonView.tiles.get('3dgs:gravity').caption.textContent.includes('min length 10 cm')`);
    console.log(JSON.stringify({centimeterFilter:{...cmFilterAfter,visibleSkeletonCm:cmFilterBefore.length}}));
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        l.comparisonView.root.querySelector('.compare-skeleton').scrollIntoView({block:'start'});return true;})()`);
    const djiShot=await cdp.send('Page.captureScreenshot',{format:'png'});
    await writeFile('/tmp/depth_comparison_dji_skeleton.png',Buffer.from(djiShot.data,'base64'));
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        l.comparisonView.root.querySelector('.compare-final').scrollIntoView({block:'center'});return true;})()`);
    const finalShot=await cdp.send('Page.captureScreenshot',{format:'png'});
    await writeFile('/tmp/depth_comparison_dji_final.png',Buffer.from(finalShot.data,'base64'));

    // Save every displayed figure, including scrolled-away stages, without
    // re-inference. Pause first and hold the folder dialog to test frozen pixels.
    const figureStart=await evaluate(`(async()=>{
        const p=scene.events.invoke('annotation.panel'),l=p.liveView;
        const toggle=document.querySelector('.live-controls input[type="checkbox"]');
        toggle.checked=false;toggle.dispatchEvent(new Event('change'));
        window._qaFigureToggle=toggle;window._qaFigureFiles=[];
        window._qaFigureHash=async canvas=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
            canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data))).map(x=>x.toString(16).padStart(2,'0')).join('');
        window._qaFigureExpected={};
        for(const figure of l.collectFigures()) window._qaFigureExpected[figure.name+'.png']={
            hash:await window._qaFigureHash(figure.canvas),width:figure.canvas.width,height:figure.canvas.height};
        window._qaFigureChoose=l.chooseOutputFolder;window._qaFigureWrite=l.writeFile;
        l.chooseOutputFolder=()=>new Promise(resolve=>window._qaFigureResolve=resolve);
        p.outputDirHandle=null;
        const before={frame:l.last.frameId,calls:window._qaDepthCalls,expected:Object.keys(window._qaFigureExpected),
            disabled:l.figuresButton.disabled,paused:!l.enabled};
        l.figuresButton.click();return before;
    })()`);
    assert.equal(figureStart.paused,true);assert.equal(figureStart.disabled,false);
    assert.equal(figureStart.expected.length,19);
    await waitFor('window._qaFigureResolve');
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        l.rgbCanvas.getContext('2d').fillStyle='red';l.rgbCanvas.getContext('2d').fillRect(0,0,640,640);
        const canvas=l.comparisonView.tiles.get('3dgs:skeleton').canvas;
        canvas.getContext('2d').clearRect(0,0,640,640);
        const directory=name=>({name,getDirectoryHandle:async child=>directory(name+'/'+child),
            getFileHandle:async file=>({createWritable:async()=>({write:async blob=>window._qaFigureFiles.push({name:file,path:name+'/'+file,blob}),close:async()=>{}})})});
        window._qaFigureResolve(directory('qa_figures'));return true;
    })()`);
    await waitFor('window._qaFigureFiles.length===20 && !scene.events.invoke("annotation.panel").liveView.savingFigures');
    const figures=await evaluate(`(async()=>{const failed=[];let overview;
        for(const file of window._qaFigureFiles){
            const image=await createImageBitmap(file.blob);
            const expected=window._qaFigureExpected[file.name];
            if(expected){const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
                canvas.getContext('2d').drawImage(image,0,0);
                if(image.width!==expected.width||image.height!==expected.height||await window._qaFigureHash(canvas)!==expected.hash)failed.push(file.name);
            }else if(file.name==='overview.png') overview={width:image.width,height:image.height};
            image.close();
        }
        const l=scene.events.invoke('annotation.panel').liveView;
        return {count:window._qaFigureFiles.length,failed,overview,frame:l.last.frameId,calls:window._qaDepthCalls,
            names:window._qaFigureFiles.map(file=>file.name),allPng:window._qaFigureFiles.every(file=>file.blob.type==='image/png'),
            paths:window._qaFigureFiles.map(file=>file.path),button:l.figuresButton.textContent};
    })()`);
    assert.deepEqual(figures.failed,[],'PNG pixels must match the displayed canvases captured before the folder dialog');
    assert.equal(figures.allPng,true);assert.equal(figures.frame,figureStart.frame);assert.equal(figures.calls,figureStart.calls);
    assert.equal(figures.overview.width,1352);assert.ok(figures.overview.height>6400);
    assert.deepEqual(figures.names.slice(0,19),figureStart.expected);
    assert.ok(figures.paths.every(name=>name.startsWith('qa_figures/Figures/'+figureStart.frame+'_figures_')));
    assert.equal(figures.button,'Save Figures');
    const overview=await evaluate(`(async()=>{const blob=window._qaFigureFiles.find(x=>x.name==='overview.png').blob;
        const bytes=new Uint8Array(await blob.arrayBuffer());let text='';for(let i=0;i<bytes.length;i+=8192)text+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(text);})()`);
    await writeFile('/tmp/bbox_figures_overview.png',Buffer.from(overview,'base64'));
    console.log(JSON.stringify({figures}));

    // A cancelled chooser writes nothing and restores the button. Without a
    // directory API, all the PNGs travel in a single ZIP download.
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        l.rgbCanvas.getContext('2d').drawImage(l.last.rgb,0,0);l.renderDepthComparison(l.last);
        l.chooseOutputFolder=async()=>{throw new DOMException('Cancelled','AbortError')};return true;})()`);
    assert.equal(await evaluate(`scene.events.invoke('annotation.panel').liveView.saveFigures()`),null);
    assert.equal(await evaluate(`scene.events.invoke('annotation.panel').liveView.figuresButton.disabled`),false);
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;l.chooseOutputFolder=async()=>null;
        l.writeFile=async(name,blob)=>{window._qaFigureZip={name,blob};};return true;})()`);
    const zipReceipt=await evaluate(`scene.events.invoke('annotation.panel').liveView.saveFigures()`);
    assert.equal(zipReceipt.count,20);assert.match(zipReceipt.destination,/\.zip$/);
    const archive=await evaluate(`(async()=>{const bytes=new Uint8Array(await window._qaFigureZip.blob.arrayBuffer());let text='';
        for(let i=0;i<bytes.length;i+=8192)text+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(text);})()`);
    await writeFile('/tmp/bbox_figures_fallback.zip',Buffer.from(archive,'base64'));
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;l.chooseOutputFolder=window._qaFigureChoose;l.writeFile=window._qaFigureWrite;
        window._qaFigureToggle.checked=true;window._qaFigureToggle.dispatchEvent(new Event('change'));return true;})()`);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').frameId!==${JSON.stringify(figureStart.frame)}`);

    // Save keeps the previous DA2 filenames and uses that source's filtered result.
    await evaluate(`(()=>{window._qaFiles=[];const directory=name=>({name,
        getDirectoryHandle:async child=>directory(child),getFileHandle:async file=>({createWritable:async()=>({
            write:async blob=>window._qaFiles.push({path:name+'/'+file,text:blob.type==='application/json'?await blob.text():null}),close:async()=>{}})})});
        scene.events.invoke('annotation.panel').outputDirHandle=directory('root');return true;})()`);
    await evaluate(`scene.events.invoke('annotation.panel').liveView.save()`);
    const saved=await evaluate(`(()=>{const landing=window._qaFiles.find(x=>x.path.startsWith('Landing/')&&x.path.endsWith('.json'));
        return {files:window._qaFiles.length,ids:JSON.parse(landing.text).selected_ids,
            expected:scene.events.invoke('annotation.liveSnapshot').analyses.da2.finalLanding.selected.map(x=>x.id)};})()`);
    assert.equal(saved.files,12);assert.deepEqual(saved.ids,saved.expected);
    assert.deepEqual(unexpectedBrowserErrors(),[]);
    console.log(JSON.stringify({saved,errors:cdp.errors}));
} finally {
    cdp?.ws.close();
    chrome.kill('SIGTERM');
    await new Promise(resolve => chrome.once('exit', resolve));
    await rm(profile, { recursive: true, force: true });
    await rm(fixtureDir, { recursive: true, force: true });
}
