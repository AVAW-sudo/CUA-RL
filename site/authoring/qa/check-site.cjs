// 浏览器检查：章节齐全、真实轨迹的动作条与截图、Case 切换、示意图、回放子页、手机不横向溢出、资源无错误。
// 运行：先在仓库根目录 `python3 -m http.server 8000 --bind 127.0.0.1 --directory site/dist`，再 `node site/authoring/qa/check-site.cjs`。
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { join } = require('node:path');
const { tmpdir } = require('node:os');
const { mkdtempSync } = require('node:fs');
const assert = require('node:assert/strict');
const preview = process.env.CUA_SITE_PREVIEW_URL || 'http://127.0.0.1:8000';
const shots = mkdtempSync(join(tmpdir(), 'cua-site-'));

(async () => {
  const browser = await chromium.launch({executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}, reducedMotion: 'reduce'});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('response', r => { if (r.status() >= 400) errors.push(r.status() + ' ' + r.url()); });
    await page.goto(preview, {waitUntil: 'networkidle'});

    for (const id of ['bench', 'walk', 'grader', 'grpo', 'scalecua', 'cualite', 'data', 'run', 'next', 'conclude', 'landscape', 'proposal'])
      assert(await page.locator('#' + id).count(), 'missing section #' + id);
    assert.equal(await page.locator('#actStrips .scells').first().locator('i').count(), 46, 'action strip of trial 4');
    assert.equal(await page.locator('#advStrips .scells').nth(1).locator('i').count(), 50, 'advantage strip of trial 1');
    assert(await page.locator('#vcurve rect').count() >= 46, 'value curve not drawn');
    assert(await page.locator('#creditMap rect').count() > 100, 'credit map not drawn');
    for (const img of await page.locator('#landscape img').all()) {
      await img.scrollIntoViewIfNeeded();
      assert(await img.evaluate(el => el.complete ? el.naturalWidth === 1600 : new Promise(r => { el.onload = () => r(el.naturalWidth === 1600); el.onerror = () => r(false); })), 'official figure did not load');
    }
    for (const src of ['media/walk/mp3-t4-start.webp', 'media/walk/mp3-t4-step2.webp'])
      assert(await page.locator(`img[src="${src}"]`).evaluate(el => el.complete && el.naturalWidth === 1280), src);

    // Case：默认 MP3 的“选择工具”阶段，切到 Thunderbird 再切阶段
    const img0 = page.locator('#caseRun0 img');
    assert.match(await img0.getAttribute('src'), /mp3-r0-s1\.webp$/);
    assert.match(await page.locator('#caseRun0 h4').innerText(), /第 32 \/ 46 步/);
    await page.locator('.tab[data-case="1"]').click();
    assert.match(await img0.getAttribute('src'), /tb-r0-s3\.webp$/);
    assert.match(await page.locator('#caseRun1 h4').innerText(), /0\.66 分/);
    await page.locator('#caseStages .stage').nth(1).click();
    assert.match(await page.locator('#caseRun1 img').getAttribute('src'), /tb-r1-s1\.webp$/);
    await page.locator('#caseRun0 .ribbon .mk').first().click();
    assert.match(await img0.getAttribute('src'), /tb-r0-s0\.webp$/);
    assert(await img0.evaluate(el => el.complete ? el.naturalWidth > 0 : new Promise(r => { el.onload = () => r(el.naturalWidth > 0); el.onerror = () => r(false); })), 'case frame did not load');
    await page.locator('#casePlayer').screenshot({path: join(shots, 'case.jpg'), type: 'jpeg', quality: 82});

    // 回放子页
    const replay = await browser.newPage({viewport: {width: 1280, height: 900}});
    replay.on('pageerror', e => errors.push('replay: ' + e.message));
    await replay.goto(preview + '/case/2026-11-27/screen-trace-replay.html', {waitUntil: 'load'});
    assert.match(await replay.locator('#number').innerText(), /^1 \/ \d+$/);
    await replay.locator('#next').click();
    assert.match(await replay.locator('#number').innerText(), /^2 \//);
    await replay.close();

    // 手机
    await page.setViewportSize({width: 393, height: 852});
    await page.goto(preview, {waitUntil: 'networkidle'});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'horizontal overflow on phone');
    await page.screenshot({path: join(shots, 'phone.jpg'), type: 'jpeg', quality: 80});

    assert.deepEqual(errors, []);
    console.log('通过：12 个章节、三张官方图、动作条与整屏截图、value 曲线、信号分配图、Case 切换、回放子页、手机宽度、资源加载。截图在', shots);
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
