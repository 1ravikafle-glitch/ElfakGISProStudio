// == FB Unfollow V5 - fixed finder (0 cards bug) ==
// V4 scope was too narrow so it found 0. V5: find ALL dots, keep only
// small cards with avatar (follow rows), reject wide profile header.

(() => {
  if (document.getElementById('fb-unfollow-panel')) document.getElementById('fb-unfollow-panel').remove();
  document.querySelectorAll('[data-fb-hl]').forEach(el => { el.style.outline = ''; el.style.background = ''; el.removeAttribute('data-fb-hl'); });
  document.querySelectorAll('[data-fb-done]').forEach(el => el.removeAttribute('data-fb-done'));

  const S = { running: false, paused: false, stopped: false, count: 0, skipped: 0, mode: 'test', speed: 1100 };

  const panel = document.createElement('div');
  panel.id = 'fb-unfollow-panel';
  panel.style.cssText = `position:fixed;top:15px;right:15px;z-index:999999;background:#111;color:#fff;
    padding:0;border-radius:12px;font-family:Arial,sans-serif;width:310px;
    box-shadow:0 4px 20px rgba(0,0,0,.7);border:2px solid #31a24c;font-size:13px;overflow:hidden;`;
  panel.innerHTML = `
    <div id="fb-drag" style="background:#1d4023;padding:10px 14px;font-weight:bold;font-size:15px;cursor:move;user-select:none;">FB Unfollow V7 ⚡ <button id="fb-exit" style="float:right;background:#555;color:#fff;border:none;padding:3px 9px;border-radius:6px;cursor:pointer;">✖ Exit</button></div>
    <div style="padding:12px;">
    <div style="margin-bottom:6px;">Mode:
      <button id="fb-mode-test" style="background:#1877f2;color:#fff;border:none;padding:5px 10px;border-radius:6px;cursor:pointer;font-weight:bold;">TEST</button>
      <button id="fb-mode-real" style="background:#333;color:#aaa;border:1px solid #555;padding:5px 10px;border-radius:6px;cursor:pointer;">REAL</button>
    </div>
    <div style="margin-bottom:6px;">Speed:
      <button data-speed="2000" class="fb-spd" style="background:#333;color:#fff;border:1px solid #555;padding:4px 8px;border-radius:6px;cursor:pointer;">Slow</button>
      <button data-speed="1100" class="fb-spd" style="background:#31a24c;color:#fff;border:none;padding:4px 8px;border-radius:6px;cursor:pointer;">Normal</button>
      <button data-speed="500" class="fb-spd" style="background:#333;color:#fff;border:1px solid #555;padding:4px 8px;border-radius:6px;cursor:pointer;">Fast</button>
    </div>
    <div style="display:flex;gap:5px;margin-bottom:6px;">
      <button id="fb-start" style="flex:1;background:#31a24c;border:none;color:#fff;padding:8px;border-radius:6px;cursor:pointer;font-weight:bold;">▶ Start</button>
      <button id="fb-next" style="flex:1;background:#1877f2;border:none;color:#fff;padding:8px;border-radius:6px;cursor:pointer;font-weight:bold;">Next 1</button>
    </div>
    <div style="display:flex;gap:5px;margin-bottom:8px;">
      <button id="fb-pause" style="flex:1;background:#e4a11b;border:none;color:#000;padding:7px;border-radius:6px;cursor:pointer;font-weight:bold;">⏸ Pause</button>
      <button id="fb-stop" style="flex:1;background:#e41e3f;border:none;color:#fff;padding:7px;border-radius:6px;cursor:pointer;font-weight:bold;">⏹ Stop</button>
    </div>
    <button id="fb-diag" style="width:100%;background:#333;color:#ffd76a;border:1px solid #666;padding:6px;border-radius:6px;cursor:pointer;margin-bottom:6px;">🔍 Diagnose (if 0 cards)</button>
    <div id="fb-current" style="background:#222;padding:6px;border-radius:6px;margin-bottom:6px;color:#ffd76a;font-weight:bold;">Current: -</div>
    <div id="fb-status" style="background:#000;padding:8px;border-radius:6px;height:140px;overflow-y:auto;font-size:11px;white-space:pre-wrap;"></div>
    </div>`;
  document.body.appendChild(panel);

  (() => {
    const bar = document.getElementById('fb-drag');
    let sx = 0, sy = 0, ox = 0, oy = 0, drag = false;
    bar.addEventListener('mousedown', e => {
      if (e.target.id === 'fb-exit') return;
      drag = true; sx = e.clientX; sy = e.clientY;
      const r = panel.getBoundingClientRect(); ox = r.left; oy = r.top;
      panel.style.left = ox + 'px'; panel.style.top = oy + 'px'; panel.style.right = 'auto';
      e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
      if (!drag) return;
      panel.style.left = (ox + e.clientX - sx) + 'px';
      panel.style.top = Math.max(0, oy + e.clientY - sy) + 'px';
    });
    document.addEventListener('mouseup', () => drag = false);
  })();

  const $ = id => document.getElementById(id);
  const log = t => {
    const el = $('fb-status');
    if (!el) return;
    el.textContent = new Date().toLocaleTimeString() + ' ' + t + '\n' + el.textContent.slice(0, 3000);
    console.log('[FB-V5]', t);
  };
  const setCurrent = t => { const c = $('fb-current'); if (c) c.textContent = 'Current: ' + t; };
  async function sleep(ms) {
    let e = 0;
    while (e < ms) {
      if (S.stopped) return false;
      while (S.paused && !S.stopped) await new Promise(r => setTimeout(r, 300));
      if (S.stopped) return false;
      await new Promise(r => setTimeout(r, 200));
      e += 200;
    }
    return true;
  }
  const closeMenu = () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

  function getCard(dot) {
    let el = dot.parentElement;
    for (let i = 0; i < 9 && el; i++) {
      try {
        const r = el.getBoundingClientRect();
        if (el.querySelector('img') && r.width > 220 && r.width < 900) return el;
      } catch (e) {}
      el = el.parentElement;
    }
    return dot.parentElement?.parentElement || dot.parentElement;
  }

  // V5 finder: any aria-haspopup, then keep small avatar cards only
  function findDots(debug) {
    const raw = [...document.querySelectorAll('[aria-haspopup]')].filter(b => {
      if (b.closest('#fb-unfollow-panel')) return false;
      if (!b.offsetParent) return false;
      return true;
    });
    const kept = [];
    const reasons = { done: 0, noCard: 0, header: 0, small: 0 };
    for (const b of raw) {
      if (b.dataset.fbDone) { reasons.done++; continue; }
      const r = b.getBoundingClientRect();
      if (r.width < 15) { reasons.small++; continue; }
      const card = getCard(b);
      let cr = { width: 0, height: 0 };
      try { cr = card.getBoundingClientRect(); } catch (e) {}
      // header bar (Ravi Kafle top) is full-width > 800px -> reject
      if (cr.width > 780) { reasons.header++; continue; }
      // follow cards are ~250-750px wide, 60-250px tall, contain avatar img
      if (!card.querySelector('img')) { reasons.noCard++; continue; }
      if (cr.width < 200) { reasons.small++; continue; }
      b._fbCard = card;
      kept.push(b);
    }
    // sort top-to-bottom so it goes in list order, and only BELOW the Friends header
    kept.sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
    if (debug) log(`Diag: raw=${raw.length} kept=${kept.length} skipped(header=${reasons.header},done=${reasons.done},noImg=${reasons.noCard},small=${reasons.small}) widths=[${kept.slice(0,6).map(b=>Math.round(b._fbCard.getBoundingClientRect().width)).join(',')}]`);
    return kept;
  }

  function rowName(dot) {
    const card = dot._fbCard || getCard(dot);
    const texts = [...card.querySelectorAll('a, span')]
      .map(el => (el.textContent || '').trim())
      .filter(t => t.length >= 2 && t.length <= 60 && t !== '...' && !/^(following|message|like|search|friend requests|find friends)$/i.test(t));
    texts.sort((a, b) => b.length - a.length);
    return texts[0] || '(unknown)';
  }

  let lastHL = [];
  function clearHL() { lastHL.forEach(el => { try { el.style.outline = ''; el.style.background = ''; el.removeAttribute('data-fb-hl'); } catch (e) {} }); lastHL = []; }
  function highlight(dot) {
    clearHL();
    const card = dot._fbCard || getCard(dot);
    card.style.outline = '3px solid yellow';
    card.style.background = 'rgba(255,255,0,0.07)';
    card.setAttribute('data-fb-hl', '1');
    lastHL.push(card);
  }

  function visibleUnfollowEl() {
    const els = [...document.querySelectorAll('span, div')].filter(el => {
      if (el.closest('#fb-unfollow-panel')) return false;
      if (!el.offsetParent) return false;
      if (!/^\s*unfollow\s*$/i.test((el.textContent || ''))) return false;
      const r = el.getBoundingClientRect();
      if (r.width < 40 || r.width > 600 || r.height > 80) return false;
      return true;
    });
    els.sort((a, b) => a.getBoundingClientRect().width - b.getBoundingClientRect().width);
    return els.length ? (els[0].closest('div[role="button"], li') || els[0]) : null;
  }
  async function waitForUnfollow(timeoutMs = 2500) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (S.stopped) return null;
      const m = visibleUnfollowEl();
      if (m) return m;
      await new Promise(r => setTimeout(r, 150));
    }
    return null;
  }
  async function waitForConfirm(timeoutMs = 800) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (S.stopped) return null;
      const dlg = document.querySelector('div[role="dialog"]');
      if (dlg) {
        const btns = [...dlg.querySelectorAll('span, div[role="button"], button')]
          .filter(el => /^\s*unfollow\s*$/i.test((el.textContent || '').trim()) && el.offsetParent);
        if (btns.length) return btns[0].closest('[role="button"], button') || btns[0];
      }
      await new Promise(r => setTimeout(r, 150));
    }
    return null;
  }

  async function processOne() {
    let dots = findDots(false);
    if (!dots.length) {
      log('0 cards. Scrolling DOWN to load more...');
      window.scrollBy(0, 700);
      if (!(await sleep(1200))) return false;
      dots = findDots(true);
      if (!dots.length) { log('Still 0. Press 🔍 Diagnose and send me the Diag line.'); return false; }
    }
    const dot = dots[0];
    const name = rowName(dot);
    highlight(dot);
    dot.scrollIntoView({ block: 'center', behavior: 'smooth' });
    setCurrent(name);
    log(`→ "${name}"`);
    if (S.mode === 'test') {
      log(`TEST only.`); dot.dataset.fbDone = '1'; S.skipped++;
      await sleep(400); return true;
    }
    dot.click();
    log(`  ... opened, seeking Unfollow...`);
    if (!(await sleep(350))) { closeMenu(); return false; }
    const menu = await waitForUnfollow(2500);
    if (!menu) { log(`  ✖ No Unfollow. Esc+skip.`); closeMenu(); dot.dataset.fbDone = '1'; S.skipped++; await sleep(250); return true; }
    log(`  clicking Unfollow...`); menu.click();
    if (!(await sleep(400))) return false;
    const confirm = await waitForConfirm(800);
    if (confirm) { log(`  confirming...`); confirm.click(); if (!(await sleep(400))) return false; }
    else log(`  (no confirm)`);
    dot.dataset.fbDone = '1'; S.count++;
    log(`  ✅ "${name}" (total ${S.count})`);
    window.scrollBy(0, 120);
    return true;
  }

  async function loop() {
    let empty = 0;
    while (!S.stopped) {
      while (S.paused && !S.stopped) await new Promise(r => setTimeout(r, 400));
      if (S.stopped) break;
      const ok = await processOne();
      if (!ok) { empty++; if (empty >= 3) { log(`✅ Done! ${S.count} unfollowed, ${S.skipped} skipped. ✖ Exit to clean.`); break; } continue; }
      empty = 0;
      if (!(await sleep(S.speed))) break;
    }
    S.running = false;
    const st = $('fb-start'); if (st) st.textContent = '▶ Start';
  }

  $('fb-start').onclick = () => {
    if (S.running) return;
    S.running = true; S.paused = false; S.stopped = false;
    $('fb-start').textContent = '▶ Running...';
    $('fb-pause').textContent = '⏸ Pause';
    log(`STARTED ${S.mode}, gap ${S.speed / 1000}s.`);
    loop();
  };
  $('fb-next').onclick = async () => { S.stopped = false; await processOne(); };
  $('fb-pause').onclick = e => { S.paused = !S.paused; e.target.textContent = S.paused ? '▶ Resume' : '⏸ Pause'; log(S.paused ? 'Paused.' : 'Resumed.'); };
  $('fb-stop').onclick = () => { S.stopped = true; S.running = false; S.paused = false; closeMenu(); clearHL(); setCurrent('-'); $('fb-start').textContent = '▶ Start'; log(`Stopped. ${S.count} done, ${S.skipped} skipped.`); };
  $('fb-exit').onclick = () => { S.stopped = true; S.running = false; closeMenu(); clearHL(); document.querySelectorAll('[data-fb-done]').forEach(el => el.removeAttribute('data-fb-done')); panel.remove(); };
  $('fb-diag').onclick = () => { findDots(true); log('If raw=0: FB changed layout. Tell me your Brave version + send page HTML of one card (right-click ... > Inspect > copy outerHTML).'); };
  $('fb-mode-test').onclick = () => { S.mode = 'test'; $('fb-mode-test').style.background = '#1877f2'; $('fb-mode-test').style.color = '#fff'; $('fb-mode-real').style.background = '#333'; $('fb-mode-real').style.color = '#aaa'; log('TEST mode.'); };
  $('fb-mode-real').onclick = () => { if (!confirm('REAL will UNFOLLOW. Continue?')) return; S.mode = 'real'; $('fb-mode-real').style.background = '#e41e3f'; $('fb-mode-real').style.color = '#fff'; $('fb-mode-test').style.background = '#333'; $('fb-mode-test').style.color = '#aaa'; log('REAL mode.'); };
  document.querySelectorAll('.fb-spd').forEach(b => b.onclick = () => { S.speed = parseInt(b.dataset.speed); document.querySelectorAll('.fb-spd').forEach(x => x.style.background = '#333'); b.style.background = '#31a24c'; log(`Gap ${S.speed / 1000}s.`); });

  const n = findDots(true);
  log(`V7 loaded. ${n.length} cards. If 0, press 🔍 Diagnose. IMPORTANT: scroll page so follow cards are visible, then press Next 1.`);
})();
