import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

export async function withTreeDatasetBrowser(url, run, {
    flags = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
} = {}) {
    const profile = await mkdtemp(path.join(tmpdir(), 'tree-dataset-browser-'));
    const chrome = spawn(process.env.CHROME_BIN ?? '/usr/bin/google-chrome', [
        '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', ...flags,
        '--remote-debugging-port=0', `--user-data-dir=${profile}`,
        '--no-first-run', 'about:blank'
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let log = '';
    let launchError;
    chrome.stderr.on('data', chunk => { log = (log + chunk).slice(-2000); });
    chrome.on('error', error => { launchError = error; });
    let socket;
    const pending = new Map();
    try {
        let port;
        for (let attempt = 0; !port && attempt < 100; attempt++) {
            if (launchError) throw launchError;
            if (chrome.exitCode !== null) break;
            try {
                port = parseInt(await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8'), 10);
            } catch (error) {
                if (error.code !== 'ENOENT') throw error;
            }
            if (!port) await delay(100);
        }
        if (!port) throw new Error(`Chrome did not start: ${log}`);
        const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
        await once(socket, 'open');
        let nextId = 0;
        socket.addEventListener('message', event => {
            const message = JSON.parse(event.data);
            const request = pending.get(message.id);
            if (!request) return;
            pending.delete(message.id);
            clearTimeout(request.timer);
            if (message.error) request.reject(new Error(JSON.stringify(message.error)));
            else request.resolve(message.result);
        });
        socket.addEventListener('close', () => {
            for (const { timer, reject } of pending.values()) {
                clearTimeout(timer);
                reject(new Error('Browser connection closed'));
            }
            pending.clear();
        });
        const send = (method, params) => new Promise((resolve, reject) => {
            const id = ++nextId;
            const timer = setTimeout(() => {
                pending.delete(id);
                reject(new Error(`Browser timed out during ${method}`));
            }, 120000);
            pending.set(id, { resolve, reject, timer });
            socket.send(JSON.stringify({ id, method, params }));
        });
        const evaluate = async expression => {
            const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
            if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
            return result.result.value;
        };
        await send('Page.navigate', { url });
        const start = Date.now();
        while (Date.now() - start < 120000) {
            if (await evaluate("window.scene?.events?.functions.has('annotation.evaluateTreeFrame')")) {
                return await run(evaluate);
            }
            await delay(200);
        }
        throw new Error(`Annotation viewer did not initialize at ${url}`);
    } finally {
        socket?.close();
        if (chrome.pid && chrome.exitCode === null && chrome.signalCode === null) {
            const closed = once(chrome, 'exit');
            chrome.kill();
            await closed;
        }
        await rm(profile, { recursive: true, force: true, maxRetries: 4, retryDelay: 200 });
    }
}
