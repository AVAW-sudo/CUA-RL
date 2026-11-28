// RL for CUA-Agent：Case 播放器、各方法信号分配图、32 点图、章节高亮。
(function () {
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // ---------- Case 播放器（数据来自 case/2026-11-27 的关键帧对照） ----------
  var CASES = [
    {
      goal: '给 Han Baoyi 的 MP3 写入曲名 Tears of Dancing Girl 和歌手 Han Baoyi。',
      stages: ['定位目标', '选择工具', '执行操作', '结束时'], fork: 1,
      note: '同一文件四次得分 0、0、0、1。评分器重新打开 MP3，检查曲名和歌手两个标签。',
      runs: [
        {label: '试跑 4', score: '1 分', cls: 'ok', len: 46, img: 'mp3-r0',
         verdict: '后段用 Python 写入两个标签并读回，评分器读到了正确的值。',
         frames: [[2, '音乐目录列出了目标 MP3。模型双击这首歌。'], [32, '终端已列出音乐文件，接着尝试读取目标文件的现有标签。'], [41, '终端里已填好写入曲名、歌手并保存的脚本。这一步按回车执行。'], [46, '终端回读出曲名和歌手。模型结束任务，评分器从文件中读到了两个正确的标签。']]},
        {label: '试跑 1', score: '0 分', cls: 'bad', len: 50, img: 'mp3-r1',
         verdict: '后半段用在安装和等待 Kid3，到 50 步上限两个标签都没写入。',
         frames: [[12, '目标文件已经在音乐目录里。这时桌面前景是文件管理器，模型尝试输入安装命令。'], [23, '切到终端后，模型开始检查本机有没有 Kid3 标签编辑器。'], [33, '软件包已下载，终端正在解包。模型继续等待安装。'], [50, '最后一步仍在等待软件包解包。到达 50 步上限，这次没有写入正确的标签。']]}
      ]
    },
    {
      goal: '在 Thunderbird 建立 Promotions 和 Deals 两个本地文件夹，把主题含 discount 的来信自动移入 Promotions。',
      stages: ['新建目录', '输入条件', '选择去向', '结束时'], fork: -1,
      note: '两个本地目录各占 0.33 分；另 0.34 分要求规则文件里出现 discount。截图没有记录文件何时写入。',
      runs: [
        {label: '试跑 3', score: '1 分', cls: 'ok', len: 38, img: 'tb-r0',
         verdict: '两个目录和规则文件的三项检查都通过。',
         frames: [[20, '已在新目录对话框填入 Deals，这一步点击创建。'], [30, '规则编辑器里可见主题包含 discount，接下来选择移动目的地。'], [33, '动作栏已显示 Promotions on Local Folders，这一步点击确认。'], [38, '模型结束时已回到账户页。这次两个目录和规则文件的检查全部通过。']]},
        {label: '试跑 2', score: '0.66 分', cls: 'part', len: 43, img: 'tb-r1',
         verdict: '界面显示规则已启用，评分器只给两个目录计分。',
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
        var b = document.createElement('button');
        b.type = 'button'; b.className = 'stage' + (i === c.fork ? ' fork' : '');
        b.setAttribute('aria-pressed', String(i === stage));
        b.innerHTML = '<span class="k">' + (i + 1) + '</span>' + s;
        b.onclick = function () { stage = i; render(); };
        stagesBox.appendChild(b);
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
    var X0 = 230, W = 56, G = 8, H = 34, Y = [56, 112, 168, 224, 280, 336];
    var cx = function (i) { return X0 + i * (W + G); };
    var color = function (v) { return Math.abs(v) < 0.06 ? '#e3e6ec' : v > 0 ? '#1a9850' : '#d62020'; };
    var alpha = function (v) { return Math.abs(v) < 0.06 ? 1 : (0.15 + 0.85 * Math.abs(v)).toFixed(2); };
    var cell = function (x, y, w, v) { map.appendChild(el('rect', {x: x, y: y, width: w, height: H, rx: 4, fill: color(v), 'fill-opacity': alpha(v)})); };
    var names = [['GRPO（原版）', '整条轨迹一个值'], ['PPO', '每个 token 一个值'], ['Turn-PPO', '每轮操作一个值'], ['GiGPO', '整条 + 同状态的一步'], ['Turn-PPO + 续跑校正', '续跑让每轮的值更准'], ['TA-GRPO', '按段，段内靠后加权']];
    names.forEach(function (n, r) {
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
    [[0, 3, [0.2, 0.35, 0.5, 0.65]], [4, 8, [0.15, 0.25, 0.35, 0.45, 0.55]], [9, 11, [-0.4, -0.6, -0.8]]].forEach(function (s) {
      s[2].forEach(function (v, k) { cell(cx(s[0] + k), Y[5], W, v); });
      map.appendChild(el('rect', {x: cx(s[0]) - 3, y: Y[5] - 3, width: cx(s[1]) + W - cx(s[0]) + 6, height: H + 6, rx: 7, fill: 'none', stroke: '#4b5468', 'stroke-width': 1.5}));
    });
    [0, 3].forEach(function (i) {
      map.appendChild(el('rect', {x: cx(i) - 2, y: Y[3] - 2, width: W + 4, height: H + 4, rx: 6, fill: 'none', stroke: '#f0a30a', 'stroke-width': 2.5}));
      map.appendChild(el('text', {x: cx(i) + W / 2, y: Y[3] - 7, 'font-size': 11, fill: '#a8490a', 'text-anchor': 'middle'}, '同状态'));
    });
    [3, 7].forEach(function (i, k) {
      var sx = cx(i) + W / 2, sy = Y[4];
      [[18, -18], [26, -12], [30, -5]].forEach(function (d) {
        map.appendChild(el('path', {d: 'M' + sx + ' ' + sy + 'L' + (sx + d[0]) + ' ' + (sy + d[1]), stroke: '#f0a30a', 'stroke-width': 1.8}));
        map.appendChild(el('circle', {cx: sx + d[0], cy: sy + d[1], r: 2.5, fill: '#f0a30a'}));
      });
      if (k === 0) map.appendChild(el('text', {x: sx + 36, y: sy - 10, 'font-size': 11.5, fill: '#a8490a'}, '续跑 2–4 次'));
    });
    [[230, '#1a9850', '鼓励'], [310, '#d62020', '抑制']].forEach(function (l) {
      map.appendChild(el('rect', {x: l[0], y: 402, width: 14, height: 14, rx: 3, fill: l[1]}));
      map.appendChild(el('text', {x: l[0] + 20, y: 414, 'font-size': 13, fill: '#4b5468'}, l[2]));
    });
    map.appendChild(el('rect', {x: 390, y: 401, width: 16, height: 16, rx: 3, fill: 'none', stroke: '#f0a30a', 'stroke-width': 2.5}));
    map.appendChild(el('text', {x: 412, y: 414, 'font-size': 13, fill: '#4b5468'}, '同状态比较'));
    map.appendChild(el('path', {d: 'M512 414L528 402', stroke: '#f0a30a', 'stroke-width': 1.8}));
    map.appendChild(el('circle', {cx: 528, cy: 402, r: 2.5, fill: '#f0a30a'}));
    map.appendChild(el('text', {x: 536, y: 414, 'font-size': 13, fill: '#4b5468'}, '真实续跑'));
  }

  // ---------- 32 点图：首批审计里 17 道给关键错例满分 ----------
  var dots = document.getElementById('dots32');
  if (dots) for (var d = 0; d < 32; d++) { var s = document.createElement('i'); if (d < 17) s.className = 'r'; dots.appendChild(s); }

  // ---------- 章节高亮 ----------
  var links = [].slice.call(document.querySelectorAll('.toplinks a'));
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id); });
      });
    }, {rootMargin: '-40% 0px -55% 0px'});
    links.forEach(function (a) { var s = document.querySelector(a.getAttribute('href')); if (s) io.observe(s); });
  }
})();
