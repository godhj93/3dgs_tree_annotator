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
        document.querySelector('[aria-label="Enable bird space filter"]').click();
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
    assert.equal(synthetic.canvases,18);
    assert.ok(synthetic.different>0,'controlled DA2 step must change the skeleton before branch filtering');
    assert.equal(synthetic.outside,0);
    assert.deepEqual(synthetic.sourceFrames,[synthetic.frame,synthetic.frame]);
    assert.deepEqual(synthetic.units,['relative_pseudo3d','approximate_meters']);
    assert.equal(synthetic.overlay,'3dgs');
    console.log(JSON.stringify({synthetic}));

    const stageSettings=await evaluate(`(()=>{
        const l=scene.events.invoke('annotation.panel').liveView,landing=l.comparisonView.landing;
        const space=landing.querySelector('fieldset');
        return {
            depth:!!document.querySelector('.compare-boundary [aria-label="Shared skeleton depth gap"]'),
            merge:!!l.pcaAngleNumber.closest('.compare-merged'),
            gravity:[l.gravityAngleNumber,l.gravityLengthNumber,l.gsLengthNumber,l.gravityComponentCheckbox,l.gravityTrunkCheckbox]
                .every(input=>input.closest('.compare-gravity')),
            geometry:[l.finalWidthSlider,l.finalBearingSlider,l.finalStrictCameraCheckbox,l.finalFoliageCheckbox]
                .every(input=>landing.querySelector('details').contains(input)),
            topK:!!l.finalTopKInput.closest('.compare-final'),
            spaceVisible:space.getBoundingClientRect().height>0 && space.querySelector('input[type="number"]').getBoundingClientRect().height>0,
            spaceValues:Object.fromEntries([...space.querySelectorAll('input[type="number"]')].map(input=>[input.name,input.valueAsNumber])),
            optionalOn:[...l.comparisonView.root.querySelectorAll(':is(.compare-gravity,.compare-final,fieldset,details) input[type="checkbox"]')]
                .filter(input=>input.getAttribute('aria-label')!=='Enable bird space filter').every(input=>input.checked),
            cameraAngles:[l.finalBearingSlider,l.finalEndOnSlider,l.finalImageTiltSlider].map(input=>input.valueAsNumber)
        };
    })()`);
    assert.deepEqual(stageSettings,{depth:true,merge:true,gravity:true,geometry:true,topK:true,spaceVisible:true,optionalOn:true,
        cameraAngles:[30,90,30],spaceValues:{widthM:10,lengthM:10,heightM:10,marginM:5,approachM:1,minApproachAngleDeg:60,da2ReferenceM:null}});

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
    await evaluate(`document.querySelector('[aria-label="Enable bird space filter"]').click()`);
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
    await waitFor('scene.events.invoke("annotation.panel").liveView.spaceReady()');
    const noCandidates=await evaluate(`['da2','3dgs'].every(k=>scene.events.invoke('annotation.panel').liveView.comparisonView.tiles.get(k+':after_space').caption.textContent.includes('공간 검사에 들어온 후보 0개'))`);
    assert.equal(noCandidates,true);
    await evaluate(`(async()=>{const p=scene.events.invoke('annotation.panel'),picked=await scene.camera.intersect(.5,.72);
        for(const input of p.liveView.comparisonView.root.querySelectorAll(':is(.compare-gravity,.compare-final,fieldset,details) input[type="checkbox"]')){
            if(input.checked&&input.getAttribute('aria-label')!=='Enable bird space filter')input.click();
        }
        if(!picked)throw new Error('No DJI surface');const v=picked.position;
        p.boxes=[{id:'dji_compare',label:'branch',operation:'include',center:[v.x,v.y,v.z],size:[.6,.6,.6],euler:[0,0,0]}];
        p.box=p.boxes[0];p.activeBoxIndex=0;p.touched=true;p.writeInputs();scene.forceRender=true;return true;})()`);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').pose.segmentation.annotations[0]?.id==='dji_compare'`,240000);
    await waitFor('scene.events.invoke("annotation.panel").liveView.spaceReady()',120000);
    const boxOnly=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        return ['da2','3dgs'].map(k=>l.last.analyses[k]).map(a=>({eligible:a.finalBase.eligible.length,
            assessed:a.finalLanding.results.filter(c=>c.space?.geometry).length,
            boxOnly:a.finalLanding.results.every(c=>!c.space?.geometry?.travelLength && c.space?.approach_angle_deg==null),
            corners:a.finalLanding.results.filter(c=>c.space?.geometry).every(c=>c.space.top_corners.length===4&&c.space.geometry.low[2]===0),
            units:[...new Set(a.finalLanding.results.filter(c=>c.space?.geometry).map(c=>c.space.geometry_units))],
            worldContacts:a.finalLanding.results.filter(c=>c.space?.contact_world_m).length,
            selected:a.finalLanding.selected.length,remaining:a.finalLanding.eligible.length}));})()`);
    for(const source of boxOnly){
        assert.ok(source.eligible>3 && source.assessed>3);
        assert.equal(source.boxOnly,true);
        assert.equal(source.corners,true);
        assert.equal(source.selected,source.remaining);
    }
    assert.deepEqual(boxOnly.map(s=>s.units),[['da2_calibrated_meters'],['approximate_meters']]);
    assert.equal(boxOnly[0].worldContacts,0);
    assert.ok(boxOnly[1].worldContacts>0);
    // Space controls must also update the retained frame while Preview is paused.
    const frozenFrame=await evaluate(`(()=>{
        const l=scene.events.invoke('annotation.panel').liveView, frame=l.last.frameId;
        document.querySelector('.live-controls input[type="checkbox"]').click();
        const input=document.querySelector('input[name="widthM"]');
        input.value='1';input.dispatchEvent(new Event('change'));
        return frame;
    })()`);
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        return !l.enabled && l.spaceReady() && Object.values(l.last.analyses).every(a=>a.spaceKey.includes('"widthM":0.01'));
    })()`);
    assert.equal(await evaluate(`scene.events.invoke('annotation.liveSnapshot').frameId`),frozenFrame);
    await evaluate(`(()=>{
        const input=document.querySelector('input[name="widthM"]');
        input.value='10';input.dispatchEvent(new Event('change'));
        document.querySelector('.live-controls input[type="checkbox"]').click();
    })()`);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').frameId!==${JSON.stringify(frozenFrame)}`,120000);
    await waitFor('scene.events.invoke("annotation.panel").liveView.spaceReady()',120000);
    const previousKeys=await evaluate(`Object.fromEntries(Object.entries(scene.events.invoke('annotation.panel').liveView.last.analyses).map(([k,a])=>[k,a.spaceKey]))`);
    await evaluate(`(()=>{const input=document.querySelector('input[name="da2ReferenceM"]');input.value='2';input.dispatchEvent(new Event('change'));})()`);
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;return l.spaceReady()&&l.last.analyses.da2.spaceKey.endsWith(':2');})()`);
    assert.equal(await evaluate(`scene.events.invoke('annotation.panel').liveView.last.analyses['3dgs'].spaceKey`),previousKeys['3dgs']);
    await evaluate(`(()=>{const input=document.querySelector('input[name="da2ReferenceM"]');input.value='';input.dispatchEvent(new Event('change'));})()`);
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;return l.spaceReady()&&l.last.analyses.da2.spaceKey.endsWith(':metric');})()`);
    const spaceRendering=await evaluate(`(()=>{
        const l=scene.events.invoke('annotation.panel').liveView,root=l.last;
        const checked=Object.entries(root.analyses).map(([source,analysis])=>
            ({source,analysis,original:analysis.finalLanding,inspected:l.spaceInspectIds[source]}));
        try {
            for(const item of checked) {
                const {source,analysis,original}=item;
                const statuses=source==='da2'?['clear','unknown','blocked']:['blocked','unknown','blocked'];
                const candidates=original.results.filter(c=>c.space?.geometry).slice(0,statuses.length).map((candidate,i)=>{
                    const status=statuses[i],space=candidate.space;
                    return {...candidate,space:{...space,status,
                        top_corners:space.top_corners.map((corner,j)=>({...corner,status:j===0?status:'clear'}))}};
                });
                const eligible=candidates.filter(c=>c.space.status!=='blocked');
                const ids=new Uint32Array(original.ids.length);
                for(const candidate of eligible)for(const pixel of original.candidatePixels.get(candidate.id))ids[pixel]=candidate.id;
                analysis.finalLanding={...original,results:candidates,eligible,selected:eligible,ids};
                l.spaceInspectIds[source]=candidates[0].id;
            }
            l.renderDepthComparison(root);
            const displayMask=source=>l.visibleLanding(root.analyses[source]).result.ids;
            const overlayColors=source=>{
                const ctx=l.comparisonView.tiles.get(source+':after_space').canvas.getContext('2d');
                const pixels=ctx.getImageData(0,0,640,640).data;
                const colors=new Set();
                for(let i=0;i<pixels.length;i+=4)colors.add(pixels[i]*65536+pixels[i+1]*256+pixels[i+2]);
                return [0x45ed9b,0xffdb4d,0xff3948].map(color=>colors.has(color));
            };
            const da2=root.analyses.da2.finalLanding,gs=root.analyses['3dgs'].finalLanding;
            const clearId=da2.results[0].id;
            const difference=l.comparisonDifferenceOverlay.getContext('2d').getImageData(0,0,640,640).data;
            const blockedColors=overlayColors('3dgs');
            l.spaceInspectIds['3dgs']=gs.results[1].id;
            l.renderDepthComparison(root);
            return {da2Colors:overlayColors('da2'),blockedColors,unknownColors:overlayColors('3dgs'),
                clearOnly:displayMask('da2').every((id,pixel)=>id===(da2.ids[pixel]===clearId?clearId:0)),
                no3dgs:!displayMask('3dgs').some(Boolean),
                differenceClearOnly:displayMask('da2').every((id,pixel)=>!!id===!!difference[pixel*4+3]),
                rawUnknownRetained:gs.ids.some(id=>id===gs.results[1].id)};
        } finally {
            for(const {source,analysis,original,inspected} of checked){
                analysis.finalLanding=original;l.spaceInspectIds[source]=inspected;
            }
            l.renderDepthComparison(root);
        }
    })()`);
    assert.deepEqual(spaceRendering,{da2Colors:[true,false,false],blockedColors:[false,false,false],unknownColors:[false,false,false],
        clearOnly:true,no3dgs:true,differenceClearOnly:true,rawUnknownRetained:true});
    console.log(JSON.stringify({boxOnly}));
    // The stricter legacy pipeline remains available by opting in to its stages.
    await evaluate(`(()=>{
        for(const label of ['Enable bird space filter','Enable gravity angle filter','Enable minimum length filter',
            'Enable branch shape filter','Enable depth spread filter','Enable relative height filter','Reject green foliage',
            'Limit final landing results to Top K']) document.querySelector('[aria-label="'+label+'"]').click();
        return true;
    })()`);
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
    assert.equal(figureStart.expected.length,23);
    for(const source of ['da2','3dgs']) for(const stage of ['before_space','after_space']) {
        assert.ok(figureStart.expected.includes(`${source}_${stage}.png`));
    }
    await waitFor('window._qaFigureResolve');
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        l.rgbCanvas.getContext('2d').fillStyle='red';l.rgbCanvas.getContext('2d').fillRect(0,0,640,640);
        const canvas=l.comparisonView.tiles.get('3dgs:skeleton').canvas;
        canvas.getContext('2d').clearRect(0,0,640,640);
        const directory=name=>({name,getDirectoryHandle:async child=>directory(name+'/'+child),
            getFileHandle:async file=>({createWritable:async()=>({write:async blob=>window._qaFigureFiles.push({name:file,path:name+'/'+file,blob}),close:async()=>{}})})});
        window._qaFigureResolve(directory('qa_figures'));return true;
    })()`);
    await waitFor('window._qaFigureFiles.length===24 && !scene.events.invoke("annotation.panel").liveView.savingFigures');
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
    assert.deepEqual(figures.names.slice(0,-1),figureStart.expected);
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
    assert.equal(zipReceipt.count,24);assert.match(zipReceipt.destination,/\.zip$/);
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
    assert.equal(saved.files,14);assert.deepEqual(saved.ids,saved.expected);

    // Enable the new feature on the real DJI frame. The existing mask, depth,
    // frame ID and source-local branch IDs are reused; no inference is repeated.
    const spaceBefore=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        return {frame:l.last.frameId,calls:window._qaDepthCalls,
            captions:['da2','3dgs'].map(k=>l.comparisonView.tiles.get(k+':after_space').caption.textContent)};})()`);
    assert.ok(spaceBefore.captions.every(caption=>caption.includes('필터 꺼짐')));
    await evaluate(`document.querySelector('[aria-label="Enable bird space filter"]').click()`);
    await waitFor(`scene.events.invoke('annotation.panel').liveView.spaceReady() && !scene.events.invoke('annotation.panel').liveView.saveButton.disabled`);
    const spaces=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        return {frame:l.last.frameId,calls:window._qaDepthCalls,profile:l.birdSpace,
            sources:Object.fromEntries(['da2','3dgs'].map(k=>{const a=l.last.analyses[k];return [k,{
                pending:a.spacePending,base:a.finalBase.eligible.length,
                candidates:a.finalLanding.results.filter(c=>c.space).map(c=>({id:c.id,status:c.space.status,reasons:c.space.reasons,
                    approach:c.space.approach_angle_deg,box:c.space.geometry?.high.map((v,i)=>v-c.space.geometry.low[i])})),
                selected:a.finalLanding.selected.map(c=>c.id)}]}))};})()`);
    assert.equal(spaces.frame,spaceBefore.frame);assert.equal(spaces.calls,spaceBefore.calls);
    assert.deepEqual([spaces.profile.widthM,spaces.profile.lengthM,spaces.profile.heightM],[.1,.1,.1]);
    assert.equal(spaces.profile.checkApproachPath,false);
    assert.equal(spaces.profile.checkApproachAngle,false);
    for(const a of Object.values(spaces.sources)){
        assert.equal(a.pending,false);assert.equal(a.candidates.length,a.base);
        assert.ok(a.candidates.every(c=>['clear','unknown','blocked'].includes(c.status)));
        for(const c of a.candidates)if(c.box)assert.deepEqual(c.box.map(v=>Math.round(v*100)),[20,20,15]);
        assert.ok(a.selected.every(id=>a.candidates.find(c=>c.id===id).status!=='blocked'));
    }
    const spacePanels=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView,v=l.comparisonView;
        const jump=[...document.querySelectorAll('.live-controls button')].find(b=>b.textContent==='Finding landing branches');
        jump.click();
        return Object.fromEntries(['da2','3dgs'].map(k=>{
            const before=v.tiles.get(k+':before_space'),after=v.tiles.get(k+':after_space');
            const pixels=tile=>tile.canvas.getContext('2d').getImageData(0,0,640,640).data;
            const a=pixels(before),b=pixels(after);
            return [k,{before:before.caption.textContent,after:after.caption.textContent,
                sideBySide:before.canvas.parentElement.parentElement===after.canvas.parentElement.parentElement,
                changed:a.some((value,index)=>value!==b[index])}];
        }));})()`);
    for (const [source, data] of Object.entries(spaces.sources)) {
        const count = status => data.candidates.filter(candidate=>candidate.status===status).length;
        const view=spacePanels[source];
        assert.ok(view.sideBySide);
        assert.ok(view.before.endsWith(`후보 ${data.base}개`));
        assert.ok(view.after.includes(`통과 ${count('clear')} · 미확인 ${count('unknown')} · 탈락 ${count('blocked')}`));
        if (data.candidates.some(candidate=>candidate.reasons.includes('approach_end_on'))) assert.match(view.after,/접근각 [1-9]/);
        if (count('blocked')) assert.ok(view.changed);
    }
    assert.ok(Object.values(spacePanels).some(panel=>panel.changed));
    console.log(JSON.stringify({spacePanels}));
    const widthInput=await evaluate(`(()=>{
        const input=document.querySelector('input[name="widthM"]');
        const set=value=>{input.value=value;input.dispatchEvent(new Event('change'));};
        set('');const restored=input.valueAsNumber;set('70');
        return {restored,width:scene.events.invoke('annotation.panel').liveView.birdSpace.widthM};
    })()`);
    assert.deepEqual(widthInput,{restored:10,width:.7});
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;return l.spaceReady() && Object.values(l.last.analyses).every(a=>a.spaceKey.includes('"widthM":0.7'))})()`);
    await evaluate(`scene.events.invoke('annotation.panel').liveView.save()`);
    const spaceSaved=await evaluate(`(()=>{const files=window._qaFiles.filter(x=>x.path.startsWith('Landing/')&&x.path.endsWith('.json')).slice(-2);
        return files.map(f=>{const j=JSON.parse(f.text);return {source:j.depth_source,width:j.space_filter.widthM,selected:j.selected_ids,
            units:j.space_filter.geometry_units,method:j.space_filter.landing_check,
            hasSpace:j.candidates.filter(c=>c.space).length};});})()`);
    assert.deepEqual(spaceSaved.map(x=>x.source),['da2','3dgs']);assert.ok(spaceSaved.every(x=>x.width===.7));
    assert.deepEqual(spaceSaved.map(x=>x.units),['da2_calibrated_meters','approximate_meters']);
    assert.ok(spaceSaved.every(x=>x.method==='upper_corners'));
    console.log(JSON.stringify({spaces,spaceSaved}));
    // Add a path check while keeping the independent angle gate disabled.
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;l.viewMode='final';
        const width=document.querySelector('input[name="widthM"]');width.value='10';width.dispatchEvent(new Event('change'));
        document.querySelector('[aria-label="Check approach path"]').click();return true;})()`);
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;return l.spaceReady() && Object.values(l.last.analyses).every(a=>a.spaceKey.includes('"checkApproachPath":true'))})()`);
    const volumes=await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        l.comparisonView.landing.scrollIntoView({block:'start'});
        return Object.fromEntries(['da2','3dgs'].map(k=>[k,l.last.analyses[k].finalLanding.results.filter(c=>c.space).map(c=>({
            id:c.id,status:c.space.status,reasons:c.space.reasons,coverage:c.space.observed_fraction,
            travel:c.space.geometry?.travelLength,corners:c.space.top_corners?.length}))]));})()`);
    assert.ok(Object.values(volumes).flat().every(c=>!c.reasons.some(r=>r.startsWith('space_error')||r==='space_pending')));
    assert.ok(Object.values(volumes).every(source=>source.some(c=>c.travel>0&&c.corners===4)));
    const spaceShot=await cdp.send('Page.captureScreenshot',{format:'png'});
    await writeFile('/tmp/avian_landing_space_dji.png',Buffer.from(spaceShot.data,'base64'));
    await evaluate(`document.querySelector('[aria-label="Check minimum approach angle"]').click()`);
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;return l.spaceReady() && Object.values(l.last.analyses).every(a=>a.spaceKey.includes('"checkApproachAngle":true'))})()`);
    const angleReasons=await evaluate(`Object.values(scene.events.invoke('annotation.panel').liveView.last.analyses).flatMap(a=>a.finalLanding.results.flatMap(c=>c.space?.reasons??[]))`);
    assert.ok(angleReasons.includes('approach_end_on'));
    // Delay old space responses while new settings finish on the same frame.
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;
        window._qaHeldSpace=[];window._qaOriginalSpace=l.spaceClient.assess.bind(l.spaceClient);
        l.spaceClient.assess=(...args)=>args[6].widthM===.51?new Promise(resolve=>window._qaHeldSpace.push({resolve,base:args[5]})):window._qaOriginalSpace(...args);
        l.birdSpace.widthM=.51;l.queueComparisonRender();return true;})()`);
    await waitFor('window._qaHeldSpace.length===2');
    assert.ok(await evaluate(`['da2','3dgs'].every(k=>scene.events.invoke('annotation.panel').liveView.comparisonView.tiles.get(k+':after_space').caption.textContent.includes('공간 검사 중'))`));
    await evaluate(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;l.birdSpace.widthM=.52;l.queueComparisonRender();return true;})()`);
    await waitFor(`(()=>{const l=scene.events.invoke('annotation.panel').liveView;return l.spaceReady() && Object.values(l.last.analyses).every(a=>a.spaceKey.includes('"widthM":0.52'))})()`);
    const staleIgnored=await evaluate(`(async()=>{const l=scene.events.invoke('annotation.panel').liveView;
        for(const held of window._qaHeldSpace)held.resolve({cpuMs:0,assessments:held.base.eligible.map(c=>({id:c.id,status:'unknown',reasons:['stale_test_marker'],
            geometry:null,approach_angle_deg:null,min_clearance_m:null,contact_camera_m:null,observed_fraction:0}))});
        await Promise.resolve();await Promise.resolve();l.spaceClient.assess=window._qaOriginalSpace;
        return Object.values(l.last.analyses).every(a=>a.spaceKey.includes('"widthM":0.52') && a.finalLanding.results.every(c=>!c.space?.reasons.includes('stale_test_marker')));})()`);
    assert.equal(staleIgnored,true);
    // Exercise the real measurement button with known world-height separation.
    const height=await evaluate(`(()=>{const s=scene.getElementsByType('splat')[0],e=scene.events;
        const a=scene.camera.mainCamera.getPosition().clone().set(0,0,0),b=a.clone().set(1,2,3);
        s.measurePoints=[a,b];e.fire('selection',s);e.fire('tool.measure');
        const aw=a.clone(),bw=b.clone();s.worldTransform.transformPoint(a,aw);s.worldTransform.transformPoint(b,bw);
        const g=e.invoke('annotation.liveSnapshot').pose.gravity.vector;
        const delta=[bw.x-aw.x,bw.y-aw.y,bw.z-aw.z];const raw=Math.abs(g.reduce((sum,v,i)=>sum+v*delta[i],0));
        const button=[...document.querySelectorAll('.pcui-button')].find(x=>x.textContent==='Apply Tree Height');
        if(!button)throw new Error('height calibration control missing');button.click();
        const p=e.invoke('annotation.panel');return {expected:3/raw,scale:p.metersPerSceneUnit,reference:p.heightScaleReference,
            exported:p.payload().metric_scale.height_reference,url:new URL(location.href).searchParams.get('heightReference')};})()`);
    assert.ok(Math.abs(height.scale-height.expected)<1e-6);
    assert.equal(height.reference.actual_height_m,3);assert.deepEqual(height.exported,height.reference);
    assert.deepEqual(JSON.parse(height.url),height.reference);
    await waitFor(`${ready} && scene.events.invoke('annotation.liveSnapshot').pose.height_reference?.actual_height_m===3`);
    await waitFor(`scene.events.invoke('annotation.panel').liveView.spaceReady()`);
    assert.deepEqual(unexpectedBrowserErrors(),[]);
    console.log(JSON.stringify({volumes,height,staleIgnored}));
    assert.deepEqual(unexpectedBrowserErrors(),[]);
    console.log(JSON.stringify({saved,errors:cdp.errors}));
} finally {
    cdp?.ws.close();
    chrome.kill('SIGTERM');
    await new Promise(resolve => chrome.once('exit', resolve));
    await rm(profile, { recursive: true, force: true });
    await rm(fixtureDir, { recursive: true, force: true });
}
