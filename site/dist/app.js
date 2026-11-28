// RL for CUA-Agent：动作条、优势条、value 曲线、视觉分段格子、Case 播放器、信号分配图、章节高亮。
(function () {
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // ---------- 真实轨迹的动作序列（来自 case/2026-11-27 的逐步回放） ----------
  // l 看截图  c 点击  t 输入文字  k 按键  w 等待  s 滚动  m 移动鼠标  d 宣布完成
  var TRACE = {
    t4: 'lccccclcclsckctlcwtkccccccckctktktkwwtktktktkd',
    t1: 'lctkcwktccctktkwcccccctktktkwswswwcswwswwwswwwwcsw'
  };
  var ACT = {
    c: ['#2a78d6', '点击'], t: ['#6b4fc7', '输入文字'], k: ['#9b86e0', '按键'], w: ['#f0a30a', '等待'],
    s: ['#7a8295', '滚动'], l: ['#d5d9e1', '看截图'], m: ['#c5ccd8', '移动鼠标'], d: ['#141a29', '宣布完成']
  };
  var notePos = function (step) { return 'calc(' + (step - 1) + ' * ((100% - 98px) / 50 + 2px))'; };
  function strip(box, label, sub, codes, colorOf, keys, notes) {
    var row = document.createElement('div'); row.className = 'srow';
    if (notes && notes.length) row.style.marginBottom = '30px';
    row.innerHTML = '<span>' + label + '<small>' + sub + '</small></span>';
    var cells = document.createElement('div'); cells.className = 'scells';
    codes.split('').forEach(function (c, i) {
      var x = document.createElement('i'); var col = colorOf(c, i);
      x.style.background = col[0]; if (col[1] != null) x.style.opacity = col[1];
      if (keys && keys.indexOf(i + 1) >= 0) x.className = 'k';
      x.title = '第 ' + (i + 1) + ' 步：' + ACT[c][1];
      cells.appendChild(x);
    });
    (notes || []).forEach(function (n) {
      var s = document.createElement('span'); s.className = 'snote'; s.style.left = notePos(n[0]); s.textContent = '↑ ' + n[1]; cells.appendChild(s);
    });
    row.appendChild(cells); box.appendChild(row);
  }
  var act = document.getElementById('actStrips');
  if (act) {
    var byType = function (c) { return [ACT[c][0]]; };
    strip(act, '第 4 次 · 1 分', '46 步', TRACE.t4, byType, [2, 32, 41, 46], [[28, '第 28 步打开终端']]);
    strip(act, '第 1 次 · 0 分', '50 步，到上限', TRACE.t1, byType, [12, 23, 33, 50], [[12, '第 12 步 apt install kid3']]);
    var lg = document.createElement('div'); lg.className = 'slegend';
    'ctkwsld'.split('').forEach(function (c) { lg.innerHTML += '<span><i style="background:' + ACT[c][0] + '"></i>' + ACT[c][1] + '</span>'; });
    lg.innerHTML += '<span><i style="background:#fff;outline:2px solid #141a29"></i>下方截图对应的关键步</span>';
    act.appendChild(lg);
  }
  var adv = document.getElementById('advStrips');
  if (adv) {
    strip(adv, '第 4 次 · 优势 +0.75', '46 步都是 +0.75', TRACE.t4, function () { return ['#1a9850', 0.75]; }, null, [[3, '第 3–27 步在绕路，也被鼓励'], [41, '第 41 步改标签']]);
    strip(adv, '第 1 次 · 优势 −0.25', '50 步都是 −0.25', TRACE.t1, function () { return ['#d62020', 0.4]; }, null, [[12, '第 12 步决定装 Kid3，和其他步同样处理']]);
  }

  // ---------- value 曲线（示意） ----------
  var vc = document.getElementById('vcurve');
  if (vc) {
    var X0 = 40, CW = 920 / 50, Y0 = 128, YH = 100;
    var v = [];
    for (var s = 1; s <= 46; s++) {
      v.push(s <= 27 ? 0.25 + 0.02 * Math.sin(s) : s <= 29 ? 0.3 : s <= 31 ? 0.45 : s <= 40 ? 0.55 : s <= 45 ? 0.92 : 1);
    }
    vc.appendChild(el('text', {x: X0, y: 14, 'font-size': 12.5, fill: '#7a8295'}, 'critic 估计的 value（从这一步继续，预计能得几分）'));
    vc.appendChild(el('path', {d: 'M' + X0 + ' ' + Y0 + 'H' + (X0 + 920), stroke: '#e3e6ec'}));
    vc.appendChild(el('path', {d: 'M' + X0 + ' ' + (Y0 - YH) + 'H' + (X0 + 920), stroke: '#e3e6ec', 'stroke-dasharray': '3 4'}));
    vc.appendChild(el('text', {x: X0 - 6, y: Y0 + 4, 'font-size': 11.5, fill: '#7a8295', 'text-anchor': 'end'}, '0'));
    vc.appendChild(el('text', {x: X0 - 6, y: Y0 - YH + 4, 'font-size': 11.5, fill: '#7a8295', 'text-anchor': 'end'}, '1'));
    var d = '';
    v.forEach(function (val, i) { var x = X0 + i * CW + CW / 2, y = Y0 - YH * val; d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1); });
    vc.appendChild(el('path', {d: d, fill: 'none', stroke: '#2a78d6', 'stroke-width': 2.5}));
    TRACE.t4.split('').forEach(function (c, i) {
      vc.appendChild(el('rect', {x: X0 + i * CW + 1, y: Y0 + 12, width: CW - 2, height: 16, rx: 2, fill: ACT[c][0]}));
    });
    [[28, '打开终端'], [41, '执行改标签的脚本'], [46, '完成']].forEach(function (n, k) {
      var x = X0 + (n[0] - 1) * CW + CW / 2;
      vc.appendChild(el('path', {d: 'M' + x + ' ' + (Y0 - YH * v[n[0] - 1] - 6) + 'V' + (Y0 + 10), stroke: '#9aa6bb', 'stroke-dasharray': '2 3'}));
      vc.appendChild(el('text', {x: x, y: 172 + (k % 2) * 14, 'font-size': 12, fill: '#4b5468', 'text-anchor': k === 2 ? 'end' : 'middle'}, '第 ' + n[0] + ' 步：' + n[1]));
    });
    vc.appendChild(el('text', {x: X0, y: 172, 'font-size': 12, fill: '#7a8295'}, '第 1 步'));
  }

  // ---------- Visual Context Segmentation 示意格子 ----------
  function vcsCell(g, x, y, n, img) {
    g.appendChild(el('rect', img ? {x: x, y: y, width: 24, height: 26, rx: 3, fill: '#2a78d6'} : {x: x, y: y, width: 24, height: 26, rx: 3, fill: '#fff', stroke: '#7a8295', 'stroke-dasharray': '3 2'}));
    g.appendChild(el('text', {x: x + 12, y: y + 17, 'text-anchor': 'middle', 'font-size': 11, fill: img ? '#fff' : '#7a8295'}, String(n)));
  }
  var g1 = document.getElementById('vcs1'), g2 = document.getElementById('vcs2');
  if (g1 && g2) {
    for (var a = 1; a <= 10; a++) vcsCell(g1, 60 + (a - 1) * 27, 22, a, true);
    for (var b = 1; b <= 15; b++) vcsCell(g2, 60 + (b - 1) * 27, 86, b, b > 5);
  }

  // ---------- Case 播放器 ----------
  var CASES = [
    {
      goal: '给 Music 文件夹里的 Han Baoyi - Tears of Dancing Girl.mp3 补上曲名和歌手标签。',
      stages: ['定位目标', '选择工具', '执行操作', '结束时'], fork: 1,
      note: '四次得分 0、0、0、1。评分器重新打开 MP3，检查曲名和歌手两个标签。',
      runs: [
        {label: '第 4 次', score: '1 分', cls: 'ok', len: 46, img: 'mp3-r0',
         verdict: '用 Python 写入两个标签并读回确认。',
         frames: [[2, '音乐目录列出了目标 MP3。模型双击这首歌。'], [32, '终端已列出音乐文件，接着尝试读取目标文件的现有标签。'], [41, '终端里已填好写入曲名、歌手并保存的脚本。这一步按回车执行。'], [46, '终端回读出曲名和歌手。模型结束任务，评分器从文件中读到了两个正确的标签。']]},
        {label: '第 1 次', score: '0 分', cls: 'bad', len: 50, img: 'mp3-r1',
         verdict: '去装 Kid3，到 50 步上限两个标签都没写入。',
         frames: [[12, '目标文件已经在音乐目录里。这时桌面前景是文件管理器，模型尝试输入安装命令。'], [23, '切到终端后，模型开始检查本机有没有 Kid3 标签编辑器。'], [33, '软件包已下载，终端正在解包。模型继续等待安装。'], [50, '最后一步仍在等待软件包解包。到达 50 步上限，这次没有写入正确的标签。']]}
      ]
    },
    {
      goal: '在 Thunderbird 建 Promotions 和 Deals 两个本地文件夹，把主题含 discount 的来信自动移进 Promotions。',
      stages: ['新建文件夹', '输入条件', '选择去向', '结束时'], fork: -1,
      note: '两个文件夹各 0.33 分，规则文件里出现 discount 再得 0.34 分。截图没有记录文件何时写入。',
      runs: [
        {label: '第 3 次', score: '1 分', cls: 'ok', len: 38, img: 'tb-r0',
         verdict: '三项检查都通过。',
         frames: [[20, '已在新目录对话框填入 Deals，这一步点击创建。'], [30, '规则编辑器里可见主题包含 discount，接下来选择移动目的地。'], [33, '动作栏已显示 Promotions on Local Folders，这一步点击确认。'], [38, '模型结束时已回到账户页。这次两个目录和规则文件的检查全部通过。']]},
        {label: '第 2 次', score: '0.66 分', cls: 'part', len: 43, img: 'tb-r1',
         verdict: '界面显示规则已启用，评分器只给两个文件夹计分。',
         frames: [[17, '同样已填入 Deals，这一步点击创建。'], [36, '这里也能看到主题包含 discount，模型开始选择移动目的地。'], [41, '目的地同样显示 Promotions on Local Folders，这一步点击确认。'], [43, '结束前界面显示规则已启用。两个目录得分，规则文件的关键词检查未通过；仅凭录像无法确定原因。']]}
      ]
    }
  ];
  var player = document.getElementById('casePlayer');
  if (player) {
    var cur = 0, stage = 1;
    var tabs = [].slice.call(player.querySelectorAll('.tab'));
    var stagesBox = document.getElementById('caseStages');
    var figs = [document.getElementById('caseRun0'), document.getElementById('caseRun1')];
    var render = function () {
      var c = CASES[cur];
      tabs.forEach(function (t, i) { t.setAttribute('aria-selected', String(i === cur)); });
      document.getElementById('caseGoal').innerHTML = '<b>任务：</b>' + c.goal;
      document.getElementById('caseNote').textContent = c.note;
      stagesBox.innerHTML = '';
      c.stages.forEach(function (s, i) {
        var bt = document.createElement('button');
        bt.type = 'button'; bt.className = 'stage' + (i === c.fork ? ' fork' : '');
        bt.setAttribute('aria-pressed', String(i === stage));
        bt.innerHTML = '<span class="k">' + (i + 1) + '</span>' + s;
        bt.onclick = function () { stage = i; render(); };
        stagesBox.appendChild(bt);
      });
      c.runs.forEach(function (r, k) {
        var f = figs[k], fr = r.frames[stage];
        f.className = 'case-run ' + r.cls;
        f.querySelector('h4').innerHTML = '<span class="sc">' + r.score + '</span>' + r.label + ' · 第 ' + fr[0] + ' / ' + r.len + ' 步';
        var img = f.querySelector('img');
        img.src = 'media/case/' + r.img + '-s' + stage + '.webp'; img.alt = fr[1];
        f.querySelector('figcaption').textContent = fr[1];
        f.querySelector('.verdict').textContent = r.verdict;
        var rib = f.querySelector('.ribbon'); rib.innerHTML = '';
        var tr = document.createElement('div'); tr.className = 'track'; tr.style.width = (r.len / 50 * 100) + '%'; rib.appendChild(tr);
        r.frames.forEach(function (x, j) {
          var m = document.createElement('button');
          m.type = 'button'; m.className = 'mk'; m.style.left = (x[0] / 50 * 100) + '%';
          m.setAttribute('aria-pressed', String(j === stage));
          m.setAttribute('aria-label', c.stages[j] + '，第 ' + x[0] + ' 步');
          m.onclick = function () { stage = j; render(); };
          rib.appendChild(m);
        });
        var end = document.createElement('span'); end.className = 'end'; end.style.left = (r.len / 50 * 100) + '%'; end.textContent = r.len + ' 步'; rib.appendChild(end);
      });
    };
    tabs.forEach(function (t, i) { t.onclick = function () { cur = i; stage = CASES[i].fork >= 0 ? CASES[i].fork : 3; render(); }; });
    render();
  }

  // ---------- 各方法怎样把信号分到 12 轮（示意） ----------
  var map = document.getElementById('creditMap');
  if (map) {
    var MX = 230, W = 56, G = 8, H = 34, Y = [56, 112, 168, 224, 280, 336];
    var cx = function (i) { return MX + i * (W + G); };
    var color = function (val) { return Math.abs(val) < 0.06 ? '#e3e6ec' : val > 0 ? '#1a9850' : '#d62020'; };
    var alpha = function (val) { return Math.abs(val) < 0.06 ? 1 : (0.15 + 0.85 * Math.abs(val)).toFixed(2); };
    var cell = function (x, y, w, val) { map.appendChild(el('rect', {x: x, y: y, width: w, height: H, rx: 4, fill: color(val), 'fill-opacity': alpha(val)})); };
    [['GRPO（原版）', '整条轨迹一个值'], ['PPO', '每个 token 一个值'], ['Turn-PPO', '每轮一个值'], ['GiGPO', '整条 + 相同中间状态'], ['真实续跑（我们）', '续跑让每轮的值更准'], ['TA-GRPO', '按段，段内靠后加权']].forEach(function (n, r) {
      map.appendChild(el('text', {x: 0, y: Y[r] + 15, 'font-size': 15, 'font-weight': 600, fill: r === 4 ? '#a8490a' : '#141a29'}, n[0]));
      map.appendChild(el('text', {x: 0, y: Y[r] + 33, 'font-size': 12.5, fill: '#7a8295'}, n[1]));
    });
    map.appendChild(el('text', {x: cx(0) + W / 2, y: 30, 'font-size': 12.5, fill: '#7a8295', 'text-anchor': 'middle'}, '第 1 轮'));
    map.appendChild(el('text', {x: cx(3) + W / 2, y: 30, 'font-size': 13, fill: '#1a9850', 'font-weight': 600, 'text-anchor': 'middle'}, '第 4 轮：找对页面'));
    map.appendChild(el('text', {x: 990, y: 30, 'font-size': 13, fill: '#d62020', 'font-weight': 600, 'text-anchor': 'end'}, '第 12 轮：没保存 → 0 分'));
    [3, 11].forEach(function (i) { map.appendChild(el('path', {d: 'M' + (cx(i) + W / 2) + ' 40V378', stroke: '#c5ccd8', 'stroke-dasharray': '3 4'})); });
    var turnV = function (i) { return i === 3 ? 0.9 : i === 11 ? -0.9 : (i >= 4 && i <= 10) ? 0.15 : 0; };
    for (var i = 0; i < 12; i++) {
      cell(cx(i), Y[0], W, -0.5);
      var tok = i === 3 ? [0.5, 0.85, 0.65, 0.4] : i === 11 ? [-0.4, -0.75, -0.9, -0.6] : [[0.05, -0.1, 0.12, 0], [-0.08, 0.04, 0.1, -0.12]][i % 2];
      for (var j = 0; j < 4; j++) cell(cx(i) + j * 14.5, Y[1], 12.5, tok[j]);
      cell(cx(i), Y[2], W, turnV(i));
      cell(cx(i), Y[3], W, i === 3 ? 0.7 : -0.3);
      cell(cx(i), Y[4], W, turnV(i));
    }
    [[0, 3, [0.2, 0.35, 0.5, 0.65]], [4, 8, [0.15, 0.25, 0.35, 0.45, 0.55]], [9, 11, [-0.4, -0.6, -0.8]]].forEach(function (sg) {
      sg[2].forEach(function (val, k) { cell(cx(sg[0] + k), Y[5], W, val); });
      map.appendChild(el('rect', {x: cx(sg[0]) - 3, y: Y[5] - 3, width: cx(sg[1]) + W - cx(sg[0]) + 6, height: H + 6, rx: 7, fill: 'none', stroke: '#4b5468', 'stroke-width': 1.5}));
    });
    [0, 3].forEach(function (i2) {
      map.appendChild(el('rect', {x: cx(i2) - 2, y: Y[3] - 2, width: W + 4, height: H + 4, rx: 6, fill: 'none', stroke: '#f0a30a', 'stroke-width': 2.5}));
      map.appendChild(el('text', {x: cx(i2) + W / 2, y: Y[3] - 7, 'font-size': 11, fill: '#a8490a', 'text-anchor': 'middle'}, '相同状态'));
    });
    [3, 7].forEach(function (i3, k) {
      var sx = cx(i3) + W / 2, sy = Y[4];
      [[18, -18], [26, -12], [30, -5]].forEach(function (dd) {
        map.appendChild(el('path', {d: 'M' + sx + ' ' + sy + 'L' + (sx + dd[0]) + ' ' + (sy + dd[1]), stroke: '#f0a30a', 'stroke-width': 1.8}));
        map.appendChild(el('circle', {cx: sx + dd[0], cy: sy + dd[1], r: 2.5, fill: '#f0a30a'}));
      });
      if (k === 0) map.appendChild(el('text', {x: sx + 36, y: sy - 10, 'font-size': 11.5, fill: '#a8490a'}, '续跑 2–4 次'));
    });
    [[230, '#1a9850', '鼓励'], [310, '#d62020', '抑制']].forEach(function (l) {
      map.appendChild(el('rect', {x: l[0], y: 402, width: 14, height: 14, rx: 3, fill: l[1]}));
      map.appendChild(el('text', {x: l[0] + 20, y: 414, 'font-size': 13, fill: '#4b5468'}, l[2]));
    });
    map.appendChild(el('rect', {x: 390, y: 401, width: 16, height: 16, rx: 3, fill: 'none', stroke: '#f0a30a', 'stroke-width': 2.5}));
    map.appendChild(el('text', {x: 412, y: 414, 'font-size': 13, fill: '#4b5468'}, '相同中间状态'));
    map.appendChild(el('path', {d: 'M524 414L540 402', stroke: '#f0a30a', 'stroke-width': 1.8}));
    map.appendChild(el('circle', {cx: 540, cy: 402, r: 2.5, fill: '#f0a30a'}));
    map.appendChild(el('text', {x: 548, y: 414, 'font-size': 13, fill: '#4b5468'}, '真实续跑'));
  }

  // ---------- 章节高亮 ----------
  var links = [].slice.call(document.querySelectorAll('.toplinks a'));
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (lk) { lk.classList.toggle('on', lk.getAttribute('href') === '#' + e.target.id); });
      });
    }, {rootMargin: '-40% 0px -55% 0px'});
    links.forEach(function (lk) { var sec = document.querySelector(lk.getAttribute('href')); if (sec) io.observe(sec); });
  }
})();
