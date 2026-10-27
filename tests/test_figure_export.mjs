import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from '../supersplat/node_modules/typescript/lib/typescript.js';

const source = readFileSync(new URL('../supersplat/src/figure-export.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { figureArchive } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

const inspectZip = async archive => JSON.parse(execFileSync('python3', ['-c', `
import base64,io,json,sys,zipfile
with zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())) as archive:
    assert archive.testzip() is None
    print(json.dumps([{'name':entry.filename,'method':entry.compress_type,
        'bytes':base64.b64encode(archive.read(entry)).decode('ascii')}
        for entry in archive.infolist()]))
`], { input: Buffer.from(await archive.arrayBuffer()), encoding: 'utf8' }));

test('figure download archive preserves every PNG byte and UTF-8 filename with valid ZIP CRCs', async () => {
    const images = [Uint8Array.of(137,80,78,71,13,10,26,10,0,1,2), Uint8Array.from({length:1027}, (_,i)=>i%256)];
    const files = images.map((bytes,i)=>({name:i?'3dgs_가지.png':'overview.png',blob:new Blob([bytes],{type:'image/png'})}));
    const archive = await figureArchive(files), decoded = await inspectZip(archive);
    assert.equal(archive.type,'application/zip');
    assert.deepEqual(decoded.map(entry=>entry.name),files.map(file=>file.name));
    decoded.forEach((entry,i)=>{
        assert.equal(entry.method,0);
        assert.deepEqual(Buffer.from(entry.bytes,'base64'),Buffer.from(images[i]));
    });
});

test('empty figure ZIP is a valid empty archive', async () => {
    assert.deepEqual(await inspectZip(await figureArchive([])),[]);
});
