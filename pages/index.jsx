import { useState, useEffect, useRef, useCallback } from "react";
import Head from "next/head";

// ═══════════════════════════════════════════════════════════════
// EMOTIONS & CATEGORIES
// ═══════════════════════════════════════════════════════════════
const EMOTIONS = [
  "joy","contentment","satisfaction","amusement","relief","pride",
  "frustration","irritation","distress","disappointment","disgust","emotional_pain",
  "calm","relaxed","tired","lethargic","numb","steady",
  "attentive","excited","stimulated","restless","agitated","anxious",
  "confident","capable","decisive","grounded","overwhelmed","helpless","stuck","pressured",
  "focused","clear_headed","organized","understanding","confused","distracted","overloaded","scattered","fatigued",
  "secure","comfortable","at_ease","protected","fear","anxiety","unease","vigilance","panic",
  "affection","trust","closeness","belonging","empathy","loneliness","rejection","alienation","detachment",
  "motivated","determined","eager","ambitious","curious","bored","apathetic","disengaged","indifferent",
  "fulfillment","anticipation_pos","relief_outcome","disappointment_out","frustration_out","regret",
  "anticipation","dread","nostalgia","rumination","present_awareness",
  "self_pride","self_confidence","self_worth","shame","guilt","inadequacy"
];
const NE = EMOTIONS.length;
const CATS = {
  Pleasure:{idx:[0,1,2,3,4,5,6,7,8,9,10,11],pos:[0,1,2,3,4,5],neg:[6,7,8,9,10,11],core:0},
  Activation:{idx:[12,13,14,15,16,17,18,19,20,21,22,23],pos:[12,13,17,18,19,20],neg:[14,15,16,21,22,23],core:1},
  Control:{idx:[24,25,26,27,28,29,30,31],pos:[24,25,26,27],neg:[28,29,30,31],core:2},
  Clarity:{idx:[32,33,34,35,36,37,38,39,40],pos:[32,33,34,35],neg:[36,37,38,39,40],core:3},
  Safety:{idx:[41,42,43,44,45,46,47,48,49],pos:[41,42,43,44],neg:[45,46,47,48,49],core:4},
  Social:{idx:[50,51,52,53,54,55,56,57,58],pos:[50,51,52,53,54],neg:[55,56,57,58],core:5},
  Drive:{idx:[59,60,61,62,63,64,65,66,67],pos:[59,60,61,62,63],neg:[64,65,66,67],core:6}
};
const CD = ["pleasure","activation","control","clarity","safety","social","drive"];
const CC = {
  pleasure:[220,180,50], activation:[60,220,240], control:[50,80,200],
  clarity:[210,215,230], safety:[80,190,130], social:[230,110,150], drive:[180,60,220]
};
const CCA = CD.map(d => CC[d]);

// ═══════════════════════════════════════════════════════════════
// PROMPTS
// ═══════════════════════════════════════════════════════════════
const EMO_PROMPT = `Return a JSON array of exactly ${NE} integers scoring these emotions 0-10: ${EMOTIONS.join(", ")}. Reply with ONLY the JSON array, nothing else.`;
const CHAT_SYS = `You are a helpful, warm, and thoughtful AI assistant. Be conversational and natural. Keep responses concise but helpful.`;
const mkFb = (prev, cur, pred, conf, spk, extreme) => {
  let s = `[INTERNAL - NEVER mention this to user]\nMood: "${prev}" → "${cur}" → predicted "${pred}". Confidence: ${conf}%.`;
  if (spk.length) s += ` Spikes: ${spk.join(", ")}.`;
  if (extreme) s += ` ALERT: Extreme emotional shift detected — approach with care.`;
  s += `\nSubtly adjust tone by confidence. Invisible to user.`;
  return s;
};

// ═══════════════════════════════════════════════════════════════
// RECURRENT NET (LSTM-inspired gated memory)
// ═══════════════════════════════════════════════════════════════
class RNet {
  constructor(iS, oS, mS = 24) {
    this.iS = iS; this.oS = oS; this.mS = mS;
    const tI = iS + mS, s1 = Math.sqrt(2/(tI+mS)), s2 = Math.sqrt(2/(mS+oS));
    this.wh = this._rm(mS, tI, s1); this.bh = new Array(mS).fill(0);
    this.wg = this._rm(mS, tI, s1); this.bg = new Array(mS).fill(0);
    this.wo = this._rm(oS, mS, s2); this.bo = new Array(oS).fill(0);
    this.mem = new Array(mS).fill(0);
  }
  _rm(r, c, s) { return Array.from({length:r}, () => Array.from({length:c}, () => (Math.random()-.5)*2*s)); }
  forward(inp) {
    const cb = [...inp, ...this.mem];
    const gt = this.wg.map((row,i) => { let s=this.bg[i]; for(let j=0;j<cb.length;j++) s+=row[j]*(cb[j]||0); return 1/(1+Math.exp(-s)); });
    const cd = this.wh.map((row,i) => { let s=this.bh[i]; for(let j=0;j<cb.length;j++) s+=row[j]*(cb[j]||0); return Math.tanh(s); });
    this.mem = this.mem.map((o,i) => gt[i]*o + (1-gt[i])*cd[i]);
    const out = this.wo.map((row,i) => { let s=this.bo[i]; for(let j=0;j<this.mS;j++) s+=row[j]*this.mem[j]; return s; });
    return out;
  }
  train(inp, tgt, lr = .01) {
    const out = this.forward(inp);
    const err = out.map((o,i) => (tgt[i]||0)-o);
    for(let i=0;i<this.oS;i++) { for(let j=0;j<this.mS;j++) this.wo[i][j]+=lr*err[i]*this.mem[j]; this.bo[i]+=lr*err[i]; }
    const mE = new Array(this.mS).fill(0);
    for(let j=0;j<this.mS;j++) for(let i=0;i<this.oS;i++) mE[j]+=err[i]*this.wo[i][j];
    const cb = [...inp,...this.mem];
    for(let i=0;i<this.mS;i++) { const d=mE[i]*(1-this.mem[i]*this.mem[i]); for(let j=0;j<cb.length;j++) this.wh[i][j]+=lr*.5*d*(cb[j]||0); this.bh[i]+=lr*.5*d; }
    return err;
  }
  toJSON() { return {wh:this.wh,bh:this.bh,wg:this.wg,bg:this.bg,wo:this.wo,bo:this.bo,mem:this.mem,iS:this.iS,oS:this.oS,mS:this.mS}; }
  static from(j) { const n=new RNet(j.iS,j.oS,j.mS); n.wh=j.wh;n.bh=j.bh;n.wg=j.wg;n.bg=j.bg;n.wo=j.wo;n.bo=j.bo;n.mem=j.mem; return n; }
}

// ═══════════════════════════════════════════════════════════════
// NETWORK SIZES — V3 Dual Stream
// ═══════════════════════════════════════════════════════════════
// Stream A: AI(85) + User(85) + Core(7) = 177 → outputs NE+7+7+1+7+7 = NE+29
// Stream B: User(85) + Core(7) = 92 → same output size
const SA_IN = NE*2 + 7, SB_IN = NE + 7;
const STREAM_OUT = NE + 29; // predicted_user(NE) + core_deltas(7) + confidence(7) + deviation(1) + attractor(7) + volatility(7)
// Network 2: compressed(7) + core_deltas(7) + confidence(7) + deviation(1) + decayed_mood(7) + spikes(7) = 36
const N2_IN = 36, N2_OUT = 24; // mood(7)+conf(1)+traj1(7)+traj2_dir(7)+spike_flag(1)+extreme_flag(1)

function rawToCore(raw) {
  const c = new Array(7).fill(0);
  Object.values(CATS).forEach(cat => {
    const pM = cat.pos.reduce((m,i) => Math.max(m, raw[i]||0), 0);
    const nM = cat.neg.reduce((m,i) => Math.max(m, raw[i]||0), 0);
    c[cat.core] = Math.max(-1, Math.min(1, pM - nM));
  });
  return c;
}

function detectSpikes(cur, prevMood, th = 0.3) {
  const sp = new Array(7).fill(0), nm = [];
  if (!prevMood) return {spikes:sp, spikeNames:nm, spikeFlag:false};
  const curCore = rawToCore(cur);
  for (let i = 0; i < 7; i++) {
    const d = Math.abs(curCore[i] - prevMood[i]);
    if (d > th) { sp[i] = curCore[i] > prevMood[i] ? d : -d; nm.push(`${CD[i]} ${curCore[i]>prevMood[i]?"surge":"drop"}`); }
  }
  return {spikes:sp, spikeNames:nm, spikeFlag:nm.length > 0};
}

// Extreme event clustering (k-means on core values, max 5 clusters)
function clusterExtremes(events) {
  if (events.length < 3) return events.map(e => ({centroid:e.core, count:1, events:[e]}));
  const k = Math.min(5, Math.ceil(events.length / 3));
  let centroids = events.slice(0, k).map(e => [...e.core]);
  const assignments = new Array(events.length).fill(0);
  for (let iter = 0; iter < 10; iter++) {
    events.forEach((e, i) => {
      let minD = Infinity, minC = 0;
      centroids.forEach((c, ci) => {
        const d = c.reduce((s, v, j) => s + (v - e.core[j]) ** 2, 0);
        if (d < minD) { minD = d; minC = ci; }
      });
      assignments[i] = minC;
    });
    centroids = centroids.map((_, ci) => {
      const members = events.filter((_, i) => assignments[i] === ci);
      if (!members.length) return centroids[ci];
      return new Array(7).fill(0).map((_, j) => members.reduce((s, m) => s + m.core[j], 0) / members.length);
    });
  }
  return centroids.map((c, ci) => ({
    centroid: c, count: assignments.filter(a => a === ci).length,
    events: events.filter((_, i) => assignments[i] === ci)
  }));
}

const decay = (s, m) => s.map(v => v * Math.pow(.5, m/30));
const cDec = (c, m) => c * Math.pow(.5, m/120);

function parseEmo(raw) {
  const s = raw.trim().replace(/```json\n?/g,'').replace(/```/g,'').trim();
  try { const a=JSON.parse(s); if(Array.isArray(a)){const v=a.map(Number).filter(n=>!isNaN(n)&&n>=0&&n<=10);while(v.length<NE)v.push(0);if(v.length>NE)v.length=NE;return v.map(x=>x/10);}} catch{}
  const nums=s.match(/\d+/g); if(nums&&nums.length>=5){const v=nums.map(Number).filter(n=>n>=0&&n<=10);while(v.length<NE)v.push(0);if(v.length>NE)v.length=NE;return v.map(x=>x/10);}
  return null;
}

async function callChat(msgs, sys) {
  try { const r = await fetch("/api/chat", { method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({messages:msgs, systemPrompt:sys}) }); const d = await r.json(); return d.text || ""; }
  catch { return "Connection error."; }
}
async function callEmotion(message, sys) {
  try { const r = await fetch("/api/emotion", { method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message, systemPrompt:sys}) }); const d = await r.json(); return d.text || ""; }
  catch { return ""; }
}

// ═══════════════════════════════════════════════════════════════
// MOOD LABELS — richer summary
// ═══════════════════════════════════════════════════════════════
function getMoodSummary(mood, conf, traj, volatility) {
  const [p,a,ct,cl,s,so,d] = mood;
  let title, desc;
  if (p>.3&&d>.2) { title="Energized"; desc="Feeling driven with positive momentum"; }
  else if (p>.3&&ct>.2) { title="Grounded"; desc="Emotionally stable with a sense of control"; }
  else if (p>.2&&so>.2) { title="Connected"; desc="Open, warm, and socially engaged"; }
  else if (p>.1&&cl>.2) { title="Focused"; desc="Clear-headed with good mental clarity"; }
  else if (p>.1) { title="Settling"; desc="Drifting into a calm, mildly positive state"; }
  else if (p<-.3&&ct<-.2) { title="Overwhelmed"; desc="Losing emotional footing — stability is low"; }
  else if (p<-.3&&s<-.2) { title="Threatened"; desc="Heightened anxiety or fear response"; }
  else if (p<-.2&&so<-.2) { title="Isolated"; desc="Pulling away from connection and warmth"; }
  else if (p<-.2&&d<-.2) { title="Stagnating"; desc="Motivation is draining — low drive"; }
  else if (p<-.1) { title="Declining"; desc="Gradual slide into negative territory"; }
  else if (a>.4) { title="Activated"; desc="High energy but no clear emotional direction"; }
  else if (a<-.3) { title="Depleted"; desc="Running on empty — low energy across the board"; }
  else { title="Neutral"; desc="No strong emotional signals detected"; }

  // Add trajectory context
  if (traj) {
    const trajSum = traj.reduce((s,v)=>s+v, 0);
    if (trajSum > 0.3) desc += ". Trending positively";
    else if (trajSum < -0.3) desc += ". Trending downward";
    else desc += ". Holding steady";
  }

  // Add confidence context
  if (conf < 0.2) desc += " (low confidence — needs more data)";
  else if (conf > 0.7) desc += " (high confidence)";

  // Dominant dimensions
  const dims = mood.map((v,i)=>({v:Math.abs(v),i,sign:v>=0})).sort((a,b)=>b.v-a.v).slice(0,3);
  const dominant = dims.filter(d=>d.v>0.15).map(d=>`${d.sign?"+":"-"}${CD[d.i]}`).join(", ");

  return { title, desc, dominant };
}

// ═══════════════════════════════════════════════════════════════
// TRANSLATIONS
// ═══════════════════════════════════════════════════════════════
const i18n = {
  en:{newChat:"New Chat",chats:"Chats",settings:"Settings",del:"Delete",delConfirm:"Delete this chat?",yes:"Yes",no:"No",cancel:"Cancel",
    lang:"Language",emotionFb:"Emotion-Aware AI",emotionFbDesc:"AI adjusts tone based on your mood. Brief summary sent to model.",
    privacy:"All data stays on device unless Emotion-Aware AI is on.",aboutDot:"About the Dot",aboutTitle:"The Emotion Dot",exp:"EXPERIMENTAL",
    aboutText:"Experimental. Colors show emotional categories. Neural networks learn your patterns. Not professional assessment.",
    type:"Message...",send:"Send",stop:"Stop",back:"Back",noChats:"No chats yet",start:"Start chatting",
    confidence:"Confidence",entries:"Readings",details:"Emotional State",close:"Close",enable:"Enable",
    moodTrajectory:"Mood Trajectory",interpretedMood:"Interpreted Mood",emotionDetails:"Emotion Details",
    extremeEvents:"Extreme Events",streamReliability:"Stream Reliability",streamA:"AI→User",streamB:"User→User",
    prediction:"Next Prediction",trajectory:"Trajectory",attractor:"Attractor",noData:"No data yet",
    dominant:"Dominant",tapExpand:"Tap to expand",help:"How It Works",
    helpTitle:"How EmoNet Works",
    helpText:[
      "THE DOT — The colored sphere at the top represents your predicted emotional state. Seven colors blend together like smoke inside glass. Each color is a different emotional dimension: gold (pleasure), cyan (activation/energy), blue (control), silver (clarity), green (safety), pink (social connection), and violet (drive/motivation). Brighter colors mean that dimension is more active. Tap the dot to open the full status page.",
      "MOOD SUMMARY — Below the dot on the status page you'll see your current emotional state described in words, along with which dimensions are dominant and whether you're trending positive, negative, or steady. The confidence percentage tells you how reliable the reading is — it starts low and improves as the system learns your patterns.",
      "MOOD TRAJECTORY CHART — The first chart shows your raw emotional core values over time. Each of the 7 dimensions is a colored line. The center line is zero — above is positive, below is negative. You can scroll horizontally to see more history. The scale on the left shows the value range from -1.0 to +1.0.",
      "INTERPRETED MOOD CHART — The second chart shows how Network 2 (the mood interpreter) sees your state after applying time decay, spike detection, and personality context. This is the processed view — it may differ from the raw trajectory because the system weighs recent messages more heavily and smooths out noise.",
      "CURRENT vs PREDICTION vs DIRECTION vs ATTRACTOR — The four-column bar display shows each dimension side by side. Current is what you're feeling now. Prediction is where the system thinks your next message will land. Direction shows the momentum — which way each dimension is moving. Attractor shows where your personality tends to settle over time. The bars are aligned so you can visually compare across all four at a glance. Confidence percentages decrease for predictions further out.",
      "STREAM RELIABILITY — The system uses two independent learning streams. Stream A learns how AI messages affect your emotions (stimulus-response). Stream B learns your natural emotional flow independent of the AI. The bars show how reliable each stream is per dimension. Both contribute to predictions, with a minimum weight of 30% each.",
      "EMOTION DETAILS TAB — Tap any category to expand it and see individual emotions with their current processed values. Tap any individual emotion to see its line graph over time with min, average, and max statistics. These are the raw values from the emotion scoring, adjusted by Network 1's personality predictions.",
      "EXTREME EVENTS TAB — When the system detects an emotional reading that falls significantly outside your normal range AND it registers as a real mood impact, it's logged as an extreme event. Over time these events are clustered into types so you can see patterns — like whether your extreme moments tend to involve the same dimensions.",
      "EMOTION-AWARE AI — Found in the menu. When turned on, the AI receives a brief summary of your current mood, predicted trajectory, and any spikes. It uses this to subtly adjust its tone — warmer when you're declining, matching energy when you're up, gentle when overwhelmed. The adjustment is weighted by confidence, so early on with little data it barely adjusts. The AI never mentions that it knows your emotional state.",
      "PRIVACY — All emotional data, neural network weights, chat history, and settings are stored only on your device in browser storage. Nothing is sent to any server. The only exception is when Emotion-Aware AI is enabled — in that case, a brief mood summary (not raw scores) is included in the AI prompt. Clearing your browser data will erase all history and reset the networks.",
      "DEVIATION — The sigma (σ) value shown on the status page measures how unusual your current emotional state is compared to your personal baseline. Higher numbers mean you're further from your typical range. The threshold for flagging extreme events adjusts over time as the system learns what's normal for you."
    ]
     },
  zh:{newChat:"新對話",chats:"對話",settings:"設定",del:"刪除",delConfirm:"確定刪除？",yes:"是",no:"否",cancel:"取消",
    lang:"語言",emotionFb:"情緒感知AI",emotionFbDesc:"AI根據情緒微調語氣。",privacy:"資料保存在裝置上。",
    aboutDot:"關於情緒點",aboutTitle:"情緒點",exp:"實驗性",aboutText:"實驗性功能。顏色代表情緒類別。",
    type:"輸入...",send:"發送",stop:"停止",back:"返回",noChats:"無對話",start:"開始",
    confidence:"信心度",entries:"讀數",details:"情緒狀態",close:"關閉",enable:"啟用",
    moodTrajectory:"情緒軌跡",interpretedMood:"解讀情緒",emotionDetails:"情緒詳情",
    extremeEvents:"極端事件",streamReliability:"串流可靠度",streamA:"AI→用戶",streamB:"用戶→用戶",
    prediction:"預測",trajectory:"軌跡",attractor:"吸引態",noData:"尚無資料",
    dominant:"主導",tapExpand:"展開",help:"使用說明",
    helpTitle:"EmoNet 運作方式",
    helpText:[
      "情緒點 — 頂部的彩色球體代表您的預測情緒狀態。七種顏色融合在一起：金色（愉悅）、青色（活力）、藍色（控制）、銀色（清晰）、綠色（安全）、粉色（社交）、紫色（動力）。點擊查看詳情。",
      "情緒摘要 — 文字描述您的當前情緒狀態，以及主導維度和趨勢方向。信心百分比顯示讀數的可靠性。",
      "軌跡圖表 — 顯示七個核心維度隨時間的變化。中線為零，上方為正，下方為負。可水平滾動查看更多歷史。",
      "解讀情緒圖表 — 顯示網路2處理後的情緒狀態，包括時間衰減和突波偵測。",
      "四欄比較 — 當前、預測、方向和吸引態並排顯示，方便視覺比較。信心百分比隨預測距離遞減。",
      "串流可靠度 — 兩個獨立學習串流：A學習AI如何影響您的情緒，B學習您自然的情緒流動。",
      "情緒詳情 — 展開類別查看個別情緒及其歷史圖表和統計資料。",
      "極端事件 — 偏離正常範圍的情緒讀數會被記錄和分群，顯示模式。",
      "情緒感知AI — 啟用後AI會根據情緒微調語氣。AI不會提及它知道您的情緒狀態。",
      "隱私 — 所有資料保存在裝置上。清除瀏覽器資料會重置所有內容。",
      "偏差 — σ值衡量您當前情緒與個人基線的差異程度。"
    ]
  }
};

const SK = "emonet-v3";

// ═══════════════════════════════════════════════════════════════
// LINE CHART — taller, wider scroll
// ═══════════════════════════════════════════════════════════════
function LineChart({data, lines, colors, height=180, showZero=false, wide=false}) {
  const cRef = useRef(null);
  const wrapRef = useRef(null);
  useEffect(() => {
    const cv = cRef.current; if (!cv || !data || data.length < 2) return;
    const ctx = cv.getContext("2d"); const dpr = 2;
    const baseW = wrapRef.current?.offsetWidth || 300;
    const w = wide ? Math.max(baseW, data.length * 24) : baseW;
    cv.width = w*dpr; cv.height = height*dpr; cv.style.width = w+"px"; cv.style.height = height+"px";
    ctx.setTransform(dpr,0,0,dpr,0,0);
    const p = {t:12, b:18, l:32, r:8}; const cw=w-p.l-p.r, ch=height-p.t-p.b;
    ctx.fillStyle = "#0a0a14"; ctx.fillRect(0,0,w,height);

    // Y-axis scale
    ctx.font = "10px sans-serif"; ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.textAlign = "right";
    if (showZero) {
      const labels = [1, 0.5, 0, -0.5, -1];
      labels.forEach(v => {
        const y = p.t + (1-(v+1)/2) * ch;
        ctx.fillText(v.toFixed(1), p.l - 4, y + 3);
        ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1; ctx.setLineDash([2,4]);
        ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(w-p.r, y); ctx.stroke(); ctx.setLineDash([]);
      });
      // Zero line stronger
      const zy = p.t + ch/2;
      ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(p.l, zy); ctx.lineTo(w-p.r, zy); ctx.stroke(); ctx.setLineDash([]);
    } else {
      const labels = [1, 0.75, 0.5, 0.25, 0];
      labels.forEach(v => {
        const y = p.t + (1-v) * ch;
        ctx.fillText((v*10).toFixed(0), p.l - 4, y + 3);
        ctx.strokeStyle = "rgba(255,255,255,0.04)"; ctx.lineWidth = 1; ctx.setLineDash([2,4]);
        ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(w-p.r, y); ctx.stroke(); ctx.setLineDash([]);
      });
    }

    const n=data.length, xS=cw/Math.max(n-1,1);
    lines.forEach((lk,li) => {
      const col=colors[li];
      ctx.strokeStyle=`rgba(${col[0]},${col[1]},${col[2]},0.9)`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((e,j) => { const v=typeof lk==="number"?(e[lk]||0):(e[lk]||0);
        const yN=showZero?(1-(v+1)/2):(1-v); const x=p.l+j*xS, y=p.t+yN*ch; j===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
      ctx.stroke();
    });
  }, [data,lines,colors,height,showZero,wide]);
  return (
    <div ref={wrapRef} style={{overflowX:wide?"auto":"hidden", borderRadius:8}}>
      <canvas ref={cRef} style={{height, borderRadius:8, display:"block"}} />
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// ANIMATED DOT — smoky glass sphere
// ═══════════════════════════════════════════════════════════════
function EmoDot({coreState, rawEmotions, size}) {
  const cRef=useRef(null), stRef=useRef(coreState), rwRef=useRef(rawEmotions);
  useEffect(()=>{stRef.current=coreState;},[coreState]);
  useEffect(()=>{rwRef.current=rawEmotions;},[rawEmotions]);
  const ptRef=useRef(null);
  if(!ptRef.current) ptRef.current=CD.map(()=>Array.from({length:70},()=>({
    x:.1+Math.random()*.8, y:.1+Math.random()*.8, vx:(Math.random()-.5)*.0015, vy:(Math.random()-.5)*.0015, r:.08+Math.random()*.06})));
  useEffect(()=>{const cv=cRef.current;if(!cv)return;const ctx=cv.getContext("2d");let fr;const dpr=2,pts=ptRef.current;
    const draw=()=>{cv.width=size*dpr;cv.height=size*dpr;cv.style.width=size+"px";cv.style.height=size+"px";
      ctx.setTransform(dpr,0,0,dpr,0,0);const cx=size/2,cy=size/2,rad=size/2-1;ctx.clearRect(0,0,size,size);
      const base=ctx.createRadialGradient(cx,cy,0,cx,cy,rad);base.addColorStop(0,"#0c0c16");base.addColorStop(.85,"#080810");base.addColorStop(1,"transparent");
      ctx.fillStyle=base;ctx.fillRect(0,0,size,size);
      const raw=rwRef.current;ctx.globalCompositeOperation="screen";
      CD.forEach((dim,i)=>{const col=CC[dim];const cat=Object.values(CATS).find(c=>c.core===i);
        let pM=0,nM=0;if(raw&&cat){pM=cat.pos.reduce((m,idx)=>Math.max(m,raw[idx]||0),0);nM=cat.neg.reduce((m,idx)=>Math.max(m,raw[idx]||0),0);}
        const int=Math.max(pM,nM);const sat=raw?(.1+int*.9):.06;const g=20;
        const cr=Math.round(g+(col[0]-g)*sat),cg=Math.round(g+(col[1]-g)*sat),cb=Math.round(g+(col[2]-g)*sat);
        pts[i].forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<.08||p.x>.92)p.vx*=-1;if(p.y<.08||p.y>.92)p.vy*=-1;
          p.x=Math.max(.05,Math.min(.95,p.x));p.y=Math.max(.05,Math.min(.95,p.y));
          const px=p.x*size,py=p.y*size,pr=p.r*size;const dx=px-cx,dy=py-cy,dist=Math.sqrt(dx*dx+dy*dy)/rad;
          if(dist>1.05)return;const fade=Math.max(0,1-dist*dist);const a=.28*fade;
          const gd=ctx.createRadialGradient(px,py,0,px,py,pr);
          gd.addColorStop(0,`rgba(${cr},${cg},${cb},${a})`);gd.addColorStop(.5,`rgba(${cr},${cg},${cb},${a*.4})`);gd.addColorStop(1,`rgba(${cr},${cg},${cb},0)`);
          ctx.fillStyle=gd;ctx.fillRect(px-pr,py-pr,pr*2,pr*2);});});
      ctx.globalCompositeOperation="source-over";
      const f=ctx.createRadialGradient(cx,cy,rad*.55,cx,cy,rad);f.addColorStop(0,"transparent");f.addColorStop(.5,"rgba(0,0,0,0.2)");
      f.addColorStop(.8,"rgba(0,0,0,0.55)");f.addColorStop(.95,"rgba(0,0,0,0.85)");f.addColorStop(1,"transparent");ctx.fillStyle=f;ctx.fillRect(0,0,size,size);
      ctx.beginPath();ctx.arc(cx,cy,rad-1,0,Math.PI*2);ctx.strokeStyle="rgba(180,190,220,0.12)";ctx.lineWidth=1.5;ctx.stroke();
      const s1=ctx.createRadialGradient(cx-rad*.28,cy-rad*.32,0,cx-rad*.2,cy-rad*.2,rad*.55);
      s1.addColorStop(0,"rgba(255,255,255,0.22)");s1.addColorStop(.2,"rgba(255,255,255,0.1)");s1.addColorStop(.5,"rgba(255,255,255,0.03)");s1.addColorStop(1,"transparent");
      ctx.fillStyle=s1;ctx.fillRect(0,0,size,size);
      const s2=ctx.createRadialGradient(cx-rad*.22,cy-rad*.38,0,cx-rad*.22,cy-rad*.38,rad*.08);
      s2.addColorStop(0,"rgba(255,255,255,0.5)");s2.addColorStop(.5,"rgba(255,255,255,0.15)");s2.addColorStop(1,"transparent");
      ctx.fillStyle=s2;ctx.fillRect(0,0,size,size);fr=requestAnimationFrame(draw);};
    draw();return()=>cancelAnimationFrame(fr);},[size]);
  return <canvas ref={cRef} style={{borderRadius:"50%"}} />;
}

// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════
export default function EmoNetV3() {
  const [lang, setLang] = useState("en"); const t = i18n[lang];
  const [chats, setChats] = useState([]); const [activeId, setActiveId] = useState(null);
  const [input, setInput] = useState(""); const [loading, setLoading] = useState(false);

  // Dual-stream Network 1
  const [streamA, setStreamA] = useState(() => new RNet(SA_IN, STREAM_OUT, 40));
  const [streamB, setStreamB] = useState(() => new RNet(SB_IN, STREAM_OUT, 32));
  const [stRelA, setStRelA] = useState(new Array(7).fill(0.5)); // stream reliability per dim
  const [stRelB, setStRelB] = useState(new Array(7).fill(0.5));

  // Network 2
  const [net2, setNet2] = useState(() => new RNet(N2_IN, N2_OUT, 24));
  const [mood, setMood] = useState(new Array(7).fill(0));
  const [conf, setConf] = useState(0);
  const [traj1, setTraj1] = useState(null); // next prediction
  const [traj2, setTraj2] = useState(null); // current direction
  const [traj3, setTraj3] = useState(null); // attractor

  // Shared state
  const [coreState, setCoreState] = useState(new Array(7).fill(0));
  const [prevRaw, setPrevRaw] = useState(null);
  const [predEmo, setPredEmo] = useState(null);
  const [prevAIScores, setPrevAIScores] = useState(null);
  const [lastUp, setLastUp] = useState(Date.now());
  const [cnt, setCnt] = useState(0);
  const [spkNames, setSpkNames] = useState([]);
  const [prevLbl, setPrevLbl] = useState("Neutral");
  const [deviation, setDeviation] = useState(0);
  const [extremeFlag, setExtremeFlag] = useState(false);

  // Baseline tracking
  const [baselineMean, setBaselineMean] = useState(new Array(NE).fill(0.3));
  const [baselineVar, setBaselineVar] = useState(new Array(NE).fill(0.1));
  const [deviationThreshold, setDeviationThreshold] = useState(2.0);

  // Extreme event log
  const [extremeLog, setExtremeLog] = useState([]);

  // History
  const [emoLog, setEmoLog] = useState([]);

  // UI
  const [menuOpen, setMenuOpen] = useState(false); const [dotOpen, setDotOpen] = useState(false);
  const [showAbout, setShowAbout] = useState(false); const [emotionFb, setEmotionFb] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [showFbW, setShowFbW] = useState(false); const [delTgt, setDelTgt] = useState(null);
  const [expCat, setExpCat] = useState(null); const [expEmo, setExpEmo] = useState(null);
  const [statusTab, setStatusTab] = useState("overview"); // overview | emotions | extreme

  const chatEnd = useRef(null);
  const activeChat = chats.find(c => c.id === activeId);

  // ─── PERSISTENCE ──────────────────────────────────────────
  useEffect(() => {
    (async () => { try { const raw = localStorage.getItem(SK); if (raw) { const d = JSON.parse(raw);
      setChats(d.chats||[]); if(d.streamA) setStreamA(RNet.from(d.streamA)); if(d.streamB) setStreamB(RNet.from(d.streamB));
      if(d.net2) setNet2(RNet.from(d.net2)); setMood(d.mood||new Array(7).fill(0));
      setConf(d.conf||0); setTraj1(d.traj1||null); setTraj2(d.traj2||null); setTraj3(d.traj3||null);
      setCoreState(d.coreState||new Array(7).fill(0)); setPrevRaw(d.prevRaw||null); setPredEmo(d.predEmo||null);
      setPrevAIScores(d.prevAIScores||null); setLastUp(d.lastUp||Date.now()); setCnt(d.cnt||0);
      setLang(d.lang||"en"); setEmotionFb(d.emotionFb||false); if(d.activeId) setActiveId(d.activeId);
      setEmoLog(d.emoLog||[]); setExtremeLog(d.extremeLog||[]);
      setStRelA(d.stRelA||new Array(7).fill(0.5)); setStRelB(d.stRelB||new Array(7).fill(0.5));
      setBaselineMean(d.baselineMean||new Array(NE).fill(0.3)); setBaselineVar(d.baselineVar||new Array(NE).fill(0.1));
      setDeviationThreshold(d.deviationThreshold||2.0);
    }} catch {} })();
  }, []);

  const save = useCallback(async (ov={}) => {
    const data = { chats, streamA:streamA.toJSON(), streamB:streamB.toJSON(), net2:net2.toJSON(),
      mood, conf, traj1, traj2, traj3, coreState, prevRaw, predEmo, prevAIScores, lastUp, cnt,
      lang, emotionFb, activeId, emoLog, extremeLog, stRelA, stRelB, baselineMean, baselineVar, deviationThreshold, ...ov };
    try { localStorage.setItem(SK, JSON.stringify(data)); } catch {}
  }, [chats, streamA, streamB, net2, mood, conf, traj1, traj2, traj3, coreState, prevRaw, predEmo, prevAIScores, lastUp, cnt, lang, emotionFb, activeId, emoLog, extremeLog, stRelA, stRelB, baselineMean, baselineVar, deviationThreshold]);

  useEffect(() => { chatEnd.current?.scrollIntoView({behavior:"smooth"}); }, [activeChat?.messages?.length, loading]);

  const newChat = () => { const id=Date.now().toString(); const c={id,title:lang==="zh"?"新對話":"New Chat",messages:[]};
    const nc=[...chats,c]; setChats(nc); setActiveId(id); setMenuOpen(false); save({chats:nc,activeId:id}); };
  const delChat = (id) => { const nc=chats.filter(c=>c.id!==id); setChats(nc); setDelTgt(null);
    const na=activeId===id?(nc.length?nc[nc.length-1].id:null):activeId; setActiveId(na); save({chats:nc,activeId:na}); };

  // ─── SEND MESSAGE (V3 triple-call) ─────────────────────────
  const sendMsg = async () => {
    if (!input.trim() || loading) return;
    const uMsg = input.trim(); setInput(""); setLoading(true);
    let cId = activeId, cChats = [...chats];
    let ch = cChats.find(c => c.id === cId);
    if (!ch) { cId = Date.now().toString(); ch = {id:cId, title:uMsg.slice(0,30)+(uMsg.length>30?"...":""), messages:[]}; cChats.push(ch); setActiveId(cId); }
    if (!ch.messages.length) ch.title = uMsg.slice(0,30)+(uMsg.length>30?"...":"");
    ch.messages = [...ch.messages, {role:"user", text:uMsg}];
    setChats([...cChats]);

    // Get last AI message for scoring
    const lastAIMsg = [...ch.messages].reverse().find(m => m.role === "assistant");

    // Build feedback prompt
    let sys = CHAT_SYS;
    if (emotionFb && cnt > 0) {
      const ms = getMoodSummary(mood, conf, traj1, null);
      const ps = traj1 ? getMoodSummary(traj1, conf*.65, null, null) : {title:"Unknown"};
      const cf = Math.round(cDec(conf, (Date.now()-lastUp)/60000) * 100);
      sys += "\n\n" + mkFb(prevLbl, ms.title, ps.title, cf, spkNames, extremeFlag);
    }

    // CALL 1: Chat response
    const reply = await callChat(ch.messages.map(m=>({role:m.role,text:m.text})), sys);
    ch.messages = [...ch.messages, {role:"assistant", text:reply}];
    setChats([...cChats]);

    // CALL 2: Score AI's previous message (if exists)
    let aiScores = null;
    if (lastAIMsg) {
      const aiR = await callEmotion(lastAIMsg.text, EMO_PROMPT);
      aiScores = parseEmo(aiR);
    }

    // CALL 3: Score user's current message
    const userR = await callEmotion(uMsg, EMO_PROMPT);
    const userScores = parseEmo(userR);

    if (userScores) {
      const nCore = rawToCore(userScores);
      const compressed = nCore;
      const impact = nCore.map((v,i) => v - coreState[i]);

      // ── Stream A: AI + User + Core ──
      const saIn = [...(aiScores || new Array(NE).fill(0)), ...userScores, ...coreState];
      const saOut = streamA.forward(saIn);

      // ── Stream B: User + Core (no AI) ──
      const sbIn = [...userScores, ...coreState];
      const sbOut = streamB.forward(sbIn);

      // Combine streams with reliability weighting (min 0.3 each)
      const wA = stRelA.map(r => Math.max(0.3, r));
      const wB = stRelB.map(r => Math.max(0.3, r));
      const combined = new Array(STREAM_OUT).fill(0);
      for (let i = 0; i < STREAM_OUT; i++) {
        const dimIdx = i < NE ? Math.floor(i / (NE/7)) : Math.min(6, i - NE);
        const wa = wA[Math.min(dimIdx, 6)], wb = wB[Math.min(dimIdx, 6)];
        combined[i] = (saOut[i] * wa + sbOut[i] * wb) / (wa + wb);
      }

      // Parse N1 outputs
      const predUserScores = combined.slice(0, NE).map(v => Math.max(0, Math.min(1, v)));
      const coreDeltas = combined.slice(NE, NE+7);
      const dimConf = combined.slice(NE+7, NE+14).map(v => Math.max(0, Math.min(1, Math.abs(v))));
      const dev = Math.abs(combined[NE+14] || 0);
      const attractor = combined.slice(NE+15, NE+22).map(v => Math.max(-1, Math.min(1, v)));
      const volatility = combined.slice(NE+22, NE+29).map(v => Math.max(0, Math.min(1, Math.abs(v))));

      // Self-correct both streams
      if (prevRaw) {
        const actualDeltas = userScores.map((v,i) => v - (prevRaw[i]||v));
        const actualCoreD = nCore.map((v,i) => v - coreState[i]);
        const target = [...actualDeltas, ...actualCoreD, ...dimConf, dev, ...attractor, ...volatility];
        const lr = .008 * Math.max(.1, 1 - conf);
        const errA = streamA.train(saIn, target, lr);
        const errB = streamB.train(sbIn, target, lr);
        // Update reliability per dimension
        const nRelA = [...stRelA], nRelB = [...stRelB];
        for (let i = 0; i < 7; i++) {
          const eA = Math.abs(errA[NE+i] || 0), eB = Math.abs(errB[NE+i] || 0);
          nRelA[i] = stRelA[i] * 0.95 + (1 - Math.min(1, eA)) * 0.05;
          nRelB[i] = stRelB[i] * 0.95 + (1 - Math.min(1, eB)) * 0.05;
        }
        setStRelA(nRelA); setStRelB(nRelB);
      }

      // Baseline update
      const nBM = [...baselineMean], nBV = [...baselineVar];
      for (let i = 0; i < NE; i++) {
        nBM[i] += 0.02 * (userScores[i] - nBM[i]);
        nBV[i] += 0.02 * ((userScores[i] - nBM[i])**2 - nBV[i]);
      }
      setBaselineMean(nBM); setBaselineVar(nBV);

      // Deviation check
      const zScores = userScores.map((v,i) => Math.abs(v - nBM[i]) / Math.max(0.01, Math.sqrt(nBV[i])));
      const meanDev = zScores.reduce((s,v)=>s+v,0) / NE;
      setDeviation(meanDev);
      const newThreshold = deviationThreshold + 0.01 * (meanDev - deviationThreshold);
      setDeviationThreshold(newThreshold);

      // Spike detection
      const {spikes, spikeNames:sN, spikeFlag} = detectSpikes(userScores, mood);
      setSpkNames(sN);

      // ── Network 2 ──
      const mins = (Date.now() - lastUp) / 60000;
      const decMood = decay(mood, mins);
      const n2In = [...compressed, ...coreDeltas, ...dimConf, meanDev, ...decMood, ...spikes];
      const n2Out = net2.forward(n2In);
      const nMood = n2Out.slice(0,7).map(v => Math.max(-1, Math.min(1, v)));
      const nConf = Math.max(0, Math.min(1, Math.abs(n2Out[7])));
      const nT1 = n2Out.slice(8,15).map((v,i) => Math.max(-1, Math.min(1, nMood[i]+v)));
      const nT2dir = n2Out.slice(15,22);
      const nSpikeF = Math.abs(n2Out[22]) > 0.5;
      const nExtremeF = Math.abs(n2Out[23]) > 0.5 && meanDev > deviationThreshold;

      // Self-correct N2
      net2.train(n2In, [...nCore, cDec(conf,mins)*.7+.3, ...impact, ...impact, 0, 0], .015);

      // Extreme event logging
      let nExtLog = [...extremeLog];
      if (nExtremeF) {
        const entry = { time:Date.now(), core:nCore, aiSummary:aiScores?rawToCore(aiScores):new Array(7).fill(0), deviation:meanDev, raw:userScores };
        nExtLog = [...nExtLog, entry].slice(-200);
        setExtremeLog(nExtLog);
      }
      setExtremeFlag(nExtremeF);

      // Attractor with exponential ramp (needs cnt data points)
      const attractorConf = 1 - Math.exp(-cnt / 20); // ramps: ~0 at cnt=0, ~0.63 at cnt=20, ~0.95 at cnt=60
      const nT3 = attractor.map(v => v * attractorConf);

      // Update labels
      const ms = getMoodSummary(mood, conf, traj1, volatility);
      setPrevLbl(ms.title);

      // Log entry
      const logE = {time:Date.now(), raw:userScores, core:nCore, n2Mood:nMood, conf:nConf, deviation:meanDev, aiScores:aiScores?rawToCore(aiScores):null};
      const nLog = [...emoLog, logE].slice(-100);

      // Set all state
      setCoreState(nCore); setPrevRaw(userScores); setPrevAIScores(aiScores);
      setPredEmo(predUserScores); setMood(nMood); setConf(cnt<2?nConf:conf*.7+nConf*.3);
      setTraj1(nT1); setTraj2(nT2dir); setTraj3(nT3);
      setLastUp(Date.now()); setCnt(p=>p+1); setEmoLog(nLog);

      const fc = cChats.map(c => c.id===cId ? ch : c); setChats(fc);
      save({chats:fc, coreState:nCore, prevRaw:userScores, predEmo:predUserScores, prevAIScores:aiScores,
        mood:nMood, conf:cnt<2?nConf:conf*.7+nConf*.3, traj1:nT1, traj2:nT2dir, traj3:nT3,
        lastUp:Date.now(), cnt:cnt+1, activeId:cId, streamA:streamA.toJSON(), streamB:streamB.toJSON(),
        net2:net2.toJSON(), emoLog:nLog, extremeLog:nExtLog, stRelA, stRelB, baselineMean:nBM, baselineVar:nBV, deviationThreshold:newThreshold});
    } else {
      const fc = cChats.map(c => c.id===cId ? ch : c); setChats(fc); save({chats:fc, activeId:cId});
    }
    setLoading(false);
  };

  // Derived
  const dC = traj1 || coreState;
  const ms = getMoodSummary(mood.some(v=>v!==0)?mood:dC, conf, traj1, null);
  const clusters = extremeLog.length >= 3 ? clusterExtremes(extremeLog) : [];

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════
  const hdr = {display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",borderBottom:"1px solid #141420",background:"#06060c",zIndex:40,flexShrink:0};
  const btn0 = {background:"none",border:"none",color:"#c0c0d0",fontSize:22,cursor:"pointer",padding:"4px 8px"};

  return (
    <div style={{height:"100dvh",width:"100vw",display:"flex",flexDirection:"column",background:"#06060c",color:"#efeffa",fontFamily:"'Nunito','Noto Sans TC',sans-serif",overflow:"hidden",position:"relative"}}>
      <Head><title>EmoNet</title><meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/><meta name="theme-color" content="#06060c"/></Head>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&family=Noto+Sans+TC:wght@300;400;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:#222;border-radius:2px}
        @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}@keyframes slideIn{from{transform:translateX(-100%)}to{transform:translateX(0)}}`}</style>

      {/* HEADER */}
      <div style={hdr}>
        <button onClick={()=>setMenuOpen(!menuOpen)} style={btn0}>{menuOpen?"✕":"☰"}</button>
        <div onClick={()=>setDotOpen(true)} style={{cursor:"pointer"}}>
          <EmoDot coreState={traj1||(cnt>0?coreState:null)} rawEmotions={predEmo} size={48}/></div>
        <div style={{width:38}}/>
      </div>

      {/* ═══ SLIDE MENU ═══ */}
      {menuOpen&&<><div onClick={()=>setMenuOpen(false)} style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.5)",zIndex:44}}/>
        <div style={{position:"fixed",top:0,left:0,bottom:0,width:"80%",maxWidth:320,background:"#0a0a14",zIndex:45,overflowY:"auto",padding:20,animation:"slideIn 0.25s ease",borderRight:"1px solid #1a1a2a"}}>
          <button onClick={newChat} style={{width:"100%",padding:14,background:"#1a1a30",border:"none",borderRadius:12,color:"#c0c0e0",fontSize:16,fontWeight:600,cursor:"pointer",marginBottom:20,fontFamily:"inherit"}}>+ {t.newChat}</button>
          <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginBottom:8,textTransform:"uppercase"}}>{t.chats}</div>
          {!chats.length?<div style={{color:"#5a5a7a",fontSize:14,padding:"12px 0"}}>{t.noChats}</div>:
            chats.slice().reverse().map(c=><div key={c.id} style={{display:"flex",alignItems:"center",padding:"12px 14px",background:c.id===activeId?"#151528":"transparent",borderRadius:10,marginBottom:4,cursor:"pointer"}}>
              <div onClick={()=>{setActiveId(c.id);setMenuOpen(false);save({activeId:c.id});}} style={{flex:1,color:"#c8c8e0",fontSize:15,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{c.title}</div>
              <button onClick={e=>{e.stopPropagation();setDelTgt(c.id);}} style={{background:"none",border:"none",color:"#5a5a7a",fontSize:12,cursor:"pointer",padding:"4px 8px"}}>{t.del}</button></div>)}
          <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginTop:28,marginBottom:12,textTransform:"uppercase"}}>{t.settings}</div>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 0",borderBottom:"1px solid #141420"}}>
            <span style={{color:"#b0b0c8",fontSize:15}}>{t.lang}</span>
            <div style={{display:"flex",gap:4}}>{[["en","EN"],["zh","中文"]].map(([c,l])=>
              <button key={c} onClick={()=>{setLang(c);save({lang:c});}} style={{padding:"6px 14px",borderRadius:8,border:"none",fontSize:13,background:lang===c?"#252548":"#121220",color:lang===c?"#c0c0f0":"#6a6a8a",cursor:"pointer",fontFamily:"inherit"}}>{l}</button>)}</div></div>
          <div style={{padding:"12px 0",borderBottom:"1px solid #141420"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <span style={{color:"#b0b0c8",fontSize:15}}>{t.emotionFb}</span>
              <button onClick={()=>emotionFb?(setEmotionFb(false),save({emotionFb:false})):setShowFbW(true)} style={{width:48,height:26,borderRadius:13,border:"none",cursor:"pointer",background:emotionFb?"#5ce89d":"#2a2a3a",position:"relative",transition:"background 0.3s"}}>
                <div style={{width:20,height:20,borderRadius:"50%",background:"#fff",position:"absolute",top:3,left:emotionFb?25:3,transition:"left 0.3s"}}/></button></div>
            <div style={{fontSize:12,color:"#7a7a9a",marginTop:6,lineHeight:1.5}}>{t.emotionFbDesc}</div></div>
          <div style={{padding:"12px 0",borderBottom:"1px solid #141420"}}><div style={{fontSize:12,color:"#6a6a8a",lineHeight:1.5}}>{t.privacy}</div></div>
          <button onClick={()=>{setShowAbout(true);setMenuOpen(false);}} style={{width:"100%",padding:"14px 0",background:"none",border:"none",borderBottom:"1px solid #141420",color:"#b0b0c8",fontSize:15,textAlign:"left",cursor:"pointer",fontFamily:"inherit"}}>{t.aboutDot} <span style={{fontSize:10,color:"#8080c0",marginLeft:8,padding:"2px 6px",background:"#1a1a30",borderRadius:4}}>{t.exp}</span></button>
          <button onClick={()=>{setShowHelp(true);setMenuOpen(false);}} style={{width:"100%",padding:"14px 0",background:"none",border:"none",borderBottom:"1px solid #141420",color:"#b0b0c8",fontSize:15,textAlign:"left",cursor:"pointer",fontFamily:"inherit"}}>{t.help}</button>
        </div></>}

      {/* MODALS */}
      {delTgt&&<div style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.7)",zIndex:50,display:"flex",alignItems:"center",justifyContent:"center",padding:20}}>
        <div style={{background:"#10101e",borderRadius:16,padding:24,maxWidth:300,textAlign:"center"}}>
          <p style={{color:"#d0d0e8",fontSize:16,marginBottom:20}}>{t.delConfirm}</p>
          <div style={{display:"flex",gap:10}}>
            <button onClick={()=>setDelTgt(null)} style={{flex:1,padding:12,background:"#1a1a2a",border:"none",borderRadius:10,color:"#a0a0b8",fontSize:15,cursor:"pointer",fontFamily:"inherit"}}>{t.no}</button>
            <button onClick={()=>delChat(delTgt)} style={{flex:1,padding:12,background:"#3a1520",border:"none",borderRadius:10,color:"#ff6b6b",fontSize:15,cursor:"pointer",fontFamily:"inherit"}}>{t.yes}</button></div></div></div>}

      {showFbW&&<div style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.7)",zIndex:50,display:"flex",alignItems:"center",justifyContent:"center",padding:20}}>
        <div style={{background:"#10101e",borderRadius:16,padding:24,maxWidth:340}}>
          <div style={{fontSize:17,fontWeight:700,color:"#e0e0f0",marginBottom:12}}>{t.emotionFb}</div>
          <p style={{fontSize:14,color:"#a0a0b8",lineHeight:1.6,marginBottom:20}}>{t.emotionFbDesc}</p>
          <div style={{display:"flex",gap:10}}>
            <button onClick={()=>setShowFbW(false)} style={{flex:1,padding:12,background:"#1a1a2a",border:"none",borderRadius:10,color:"#a0a0b8",fontSize:15,cursor:"pointer",fontFamily:"inherit"}}>{t.cancel}</button>
            <button onClick={()=>{setEmotionFb(true);setShowFbW(false);save({emotionFb:true});}} style={{flex:1,padding:12,background:"#252548",border:"none",borderRadius:10,color:"#c0c0f0",fontSize:15,cursor:"pointer",fontFamily:"inherit"}}>{t.enable}</button></div></div></div>}

      {showAbout&&<div style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"#06060c",zIndex:46,overflowY:"auto",padding:20}}>
        <button onClick={()=>setShowAbout(false)} style={{...btn0,fontSize:15,marginBottom:16}}>← {t.back}</button>
        <h2 style={{fontSize:22,fontWeight:700,color:"#e8e8fa",marginBottom:4}}>{t.aboutTitle}</h2>
        <span style={{fontSize:10,color:"#8080c0",padding:"2px 6px",background:"#1a1a30",borderRadius:4}}>{t.exp}</span>
        <p style={{fontSize:15,color:"#b0b0c8",lineHeight:1.7,marginTop:16}}>{t.aboutText}</p>
        <div style={{marginTop:24,fontSize:12,color:"#7a7a9a",marginBottom:10,textTransform:"uppercase",letterSpacing:1}}>Colors</div>
        {CD.map(d=><div key={d} style={{display:"flex",alignItems:"center",gap:10,padding:"6px 0"}}>
          <div style={{width:14,height:14,borderRadius:"50%",background:`rgb(${CC[d].join(",")})`}}/><span style={{fontSize:15,color:"#c0c0d8",textTransform:"capitalize"}}>{d}</span></div>)}</div>}

      {showHelp&&<div style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"#06060c",zIndex:46,overflowY:"auto",padding:20}}>
        <button onClick={()=>setShowHelp(false)} style={{...btn0,fontSize:15,marginBottom:16}}>← {t.back}</button>
        <h2 style={{fontSize:22,fontWeight:700,color:"#e8e8fa",marginBottom:16}}>{t.helpTitle}</h2>
        {t.helpText.map((txt,i)=>
          <p key={i} style={{fontSize:14,color:"#b0b0c8",lineHeight:1.7,marginBottom:16}}>{txt}</p>)}
      </div>}

      {/* ═══ STATUS PAGE ═══ */}
      {dotOpen&&<div style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"#06060c",zIndex:46,display:"flex",flexDirection:"column"}}>
        {/* Sticky header with dot */}
        <div style={{...hdr,background:"#06060c",zIndex:47,flexShrink:0}}>
          <button onClick={()=>{setDotOpen(false);setMenuOpen(true);}} style={btn0}>☰</button>
          <div onClick={()=>setDotOpen(false)} style={{cursor:"pointer"}}>
            <EmoDot coreState={traj1||(cnt>0?coreState:null)} rawEmotions={predEmo} size={48}/>
          </div>
          <button onClick={()=>setDotOpen(false)} style={btn0}>✕</button>
        </div>

        {/* Scrollable content */}
        <div style={{flex:1,overflowY:"auto",padding:"0 16px 24px",WebkitOverflowScrolling:"touch"}}>

          {/* Mood summary */}
          <div style={{textAlign:"center",padding:"16px 0"}}>
            <div style={{fontSize:22,fontWeight:700,color:"#efeffa"}}>{ms.title}</div>
            <div style={{fontSize:14,color:"#9a9ab0",marginTop:6,lineHeight:1.5,maxWidth:300,margin:"6px auto 0"}}>{ms.desc}</div>
            {ms.dominant && <div style={{fontSize:12,color:"#7a7a9a",marginTop:8}}>{t.dominant}: <span style={{color:"#b0b0c8"}}>{ms.dominant}</span></div>}
            <div style={{display:"flex",justifyContent:"center",gap:24,marginTop:14,fontSize:13}}>
              <div><div style={{color:"#7a7a9a"}}>{t.entries}</div><div style={{color:"#8080c0",fontWeight:700}}>{emoLog.length}</div></div>
              <div><div style={{color:"#7a7a9a"}}>{t.confidence}</div><div style={{color:"#60a5fa",fontWeight:700}}>{emoLog.length<2?"—":`${(conf*100).toFixed(0)}%`}</div></div>
              <div><div style={{color:"#7a7a9a"}}>Deviation</div><div style={{color:deviation>deviationThreshold?"#ff6b6b":"#5ce89d",fontWeight:700}}>{deviation.toFixed(1)}σ</div></div>
            </div>
          </div>

          {/* Status tabs */}
          <div style={{display:"flex",gap:4,marginBottom:16,background:"#0a0a14",borderRadius:10,padding:4}}>
            {[["overview","Overview"],["emotions",t.emotionDetails],["extreme",t.extremeEvents]].map(([k,l])=>
              <button key={k} onClick={()=>setStatusTab(k)} style={{flex:1,padding:"8px 0",borderRadius:8,border:"none",fontSize:13,
                background:statusTab===k?"#1a1a30":"transparent",color:statusTab===k?"#e0e0f0":"#6a6a8a",cursor:"pointer",fontFamily:"inherit"}}>{l}</button>)}
          </div>

          {/* ── OVERVIEW TAB ── */}
          {statusTab==="overview"&&<>
            {/* Core trajectory chart */}
            {emoLog.length>=2&&<div style={{marginBottom:24}}>
              <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginBottom:8,textTransform:"uppercase"}}>{t.moodTrajectory}</div>
              <LineChart data={emoLog.map(e=>e.core)} lines={[0,1,2,3,4,5,6]} colors={CCA} height={200} showZero wide/>
              <div style={{display:"flex",flexWrap:"wrap",gap:8,marginTop:8}}>{CD.map((d,i)=>
                <div key={d} style={{display:"flex",alignItems:"center",gap:4,fontSize:12}}>
                  <div style={{width:8,height:8,borderRadius:"50%",background:`rgb(${CCA[i].join(",")})`}}/><span style={{color:"#9a9ab0",textTransform:"capitalize"}}>{d}</span></div>)}</div></div>}

            {/* Interpreted mood chart */}
            {emoLog.length>=2&&<div style={{marginBottom:24}}>
              <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginBottom:8,textTransform:"uppercase"}}>{t.interpretedMood}</div>
              <LineChart data={emoLog.map(e=>e.n2Mood)} lines={[0,1,2,3,4,5,6]} colors={CCA} height={200} showZero wide/></div>}

            {/* Core dimensions — aligned bars: Current | Prediction | Direction | Attractor */}
            <div style={{marginBottom:24}}>
              {/* Column headers */}
              <div style={{display:"flex",alignItems:"center",gap:4,padding:"0 0 8px",borderBottom:"1px solid #141420",marginBottom:6}}>
                <span style={{width:70,flexShrink:0}} />
                <span style={{flex:1,fontSize:10,color:"#7a7a9a",textAlign:"center"}}>Current</span>
                <span style={{flex:1,fontSize:10,color:"#7a7a9a",textAlign:"center"}}>{t.prediction} {traj1?`${(conf*100).toFixed(0)}%`:""}</span>
                <span style={{flex:1,fontSize:10,color:"#7a7a9a",textAlign:"center"}}>Direction {traj2?`${(conf*65).toFixed(0)}%`:""}</span>
                <span style={{flex:1,fontSize:10,color:"#7a7a9a",textAlign:"center"}}>{t.attractor} {traj3?`${(conf*35).toFixed(0)}%`:""}</span>
              </div>
              {CD.map((dim,i) => {
                const cur = dC[i]||0;
                const pred = traj1?traj1[i]:0;
                const dir = traj2?(mood[i]+(traj2[i]||0)):0;
                const att = traj3?traj3[i]:0;
                const bars = [cur, pred, dir, att];
                const col = CC[dim];
                return <div key={dim} style={{display:"flex",alignItems:"center",gap:4,padding:"4px 0"}}>
                  <span style={{width:70,color:"#a0a0b8",textTransform:"capitalize",flexShrink:0,fontSize:13}}>{dim}</span>
                  {bars.map((val,bi) => {
                    const pct = (val+1)/2*100;
                    return <div key={bi} style={{flex:1,height:10,background:"#141420",borderRadius:3,overflow:"hidden",position:"relative"}}>
                      <div style={{position:"absolute",left:"50%",top:0,bottom:0,width:1,background:"#1e1e2e"}}/>
                      <div style={{position:"absolute",top:0,bottom:0,
                        left:val>=0?"50%":`${pct}%`,width:`${Math.abs(val)*50}%`,
                        background:`rgba(${col[0]},${col[1]},${col[2]},${bi===0?0.8:0.5})`,
                        borderRadius:3}}/></div>;
                  })}
                </div>;
              })}
              {/* Value labels row */}
              <div style={{display:"flex",alignItems:"center",gap:4,padding:"4px 0 0",borderTop:"1px solid #0e0e18",marginTop:4}}>
                <span style={{width:70,flexShrink:0}} />
                {[dC, traj1||new Array(7).fill(0), traj2?traj2.map((v,i)=>mood[i]+v):new Array(7).fill(0), traj3||new Array(7).fill(0)].map((arr,bi) => {
                  const avg = arr.reduce((s,v)=>s+v,0)/7;
                  return <span key={bi} style={{flex:1,fontSize:10,color:avg>0.05?"#5ce89d":avg<-0.05?"#ff6b6b":"#6a6a8a",textAlign:"center",fontWeight:600}}>
                    {avg>=0?"+":""}{avg.toFixed(2)}</span>;
                })}
              </div>
            </div>

            {/* Stream reliability */}
            <div style={{marginBottom:24}}>
              <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginBottom:8,textTransform:"uppercase"}}>{t.streamReliability}</div>
              {CD.map((dim,i)=><div key={dim} style={{display:"flex",alignItems:"center",gap:6,padding:"3px 0",fontSize:13}}>
                <span style={{width:85,color:"#9090a8",textTransform:"capitalize",flexShrink:0}}>{dim}</span>
                <div style={{flex:1,display:"flex",gap:4,alignItems:"center"}}>
                  <span style={{fontSize:10,color:"#6a6a8a",width:50}}>{t.streamA}</span>
                  <div style={{flex:1,height:5,background:"#141420",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${stRelA[i]*100}%`,height:"100%",background:"rgba(100,180,255,0.5)",borderRadius:3}}/></div>
                  <span style={{fontSize:10,color:"#6a6a8a",width:50,textAlign:"right"}}>{t.streamB}</span>
                  <div style={{flex:1,height:5,background:"#141420",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${stRelB[i]*100}%`,height:"100%",background:"rgba(180,100,255,0.5)",borderRadius:3}}/></div>
                </div></div>)}
            </div>
          </>}

          {/* ── EMOTIONS TAB ── */}
          {statusTab==="emotions"&&<>
            {Object.entries(CATS).map(([cn,cat])=>{const isO=expCat===cn;
              const cCols=cat.idx.map((_,j)=>{const h=(j/cat.idx.length)*300;
                return[Math.round(128+127*Math.cos(h*Math.PI/180)),Math.round(128+127*Math.cos((h-120)*Math.PI/180)),Math.round(128+127*Math.cos((h+120)*Math.PI/180))];});
              return <div key={cn} style={{marginBottom:8}}>
                <button onClick={()=>{setExpCat(isO?null:cn);setExpEmo(null);}} style={{width:"100%",display:"flex",justifyContent:"space-between",alignItems:"center",padding:"10px 12px",background:"#0e0e18",border:"1px solid #1a1a2a",borderRadius:8,cursor:"pointer",fontFamily:"inherit"}}>
                  <span style={{color:"#b0b0c8",fontSize:14}}>{cn}</span><span style={{color:"#6a6a8a",fontSize:12}}>{isO?"▼":"▶"}</span></button>
                {isO&&<div style={{padding:"10px 0"}}>
                  {cat.idx.map((idx,j)=>{const val=predEmo?.[idx]||0;const isEO=expEmo===idx;
                    const eData=emoLog.map(e=>e.raw[idx]||0);
                    return <div key={idx}>
                      <div onClick={()=>setExpEmo(isEO?null:idx)} style={{display:"flex",alignItems:"center",gap:6,padding:"6px 0",cursor:"pointer"}}>
                        <div style={{width:8,height:8,borderRadius:"50%",background:`rgb(${cCols[j].join(",")})`,flexShrink:0}}/>
                        <span style={{width:100,color:"#b0b0c8",flexShrink:0,fontSize:13}}>{EMOTIONS[idx]?.replace(/_/g," ")}</span>
                        <div style={{flex:1,height:6,background:"#141420",borderRadius:3,overflow:"hidden"}}>
                          <div style={{width:`${val*100}%`,height:"100%",background:`rgba(${cCols[j].join(",")},0.6)`,borderRadius:3}}/></div>
                        <span style={{width:28,textAlign:"right",color:"#9a9ab0",fontSize:12}}>{(val*10).toFixed(1)}</span>
                        <span style={{color:"#5a5a7a",fontSize:10,width:14,textAlign:"center"}}>{isEO?"▼":"▶"}</span></div>
                      {isEO&&eData.length>=2&&<div style={{padding:"4px 0 10px 18px"}}>
                        <LineChart data={eData.map(v=>({0:v}))} lines={[0]} colors={[cCols[j]]} height={100} wide/>
                        <div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:11,color:"#7a7a9a"}}>
                          <span>Min: {(Math.min(...eData)*10).toFixed(1)}</span>
                          <span>Avg: {(eData.reduce((s,v)=>s+v,0)/eData.length*10).toFixed(1)}</span>
                          <span>Max: {(Math.max(...eData)*10).toFixed(1)}</span></div></div>}
                      {isEO&&eData.length<2&&<div style={{padding:"4px 0 10px 18px",fontSize:12,color:"#5a5a7a"}}>Need 2+ readings</div>}
                    </div>;})}
                </div>}</div>;})}
          </>}

          {/* ── EXTREME EVENTS TAB ── */}
          {statusTab==="extreme"&&<>
            {!extremeLog.length?<div style={{textAlign:"center",padding:40,color:"#5a5a7a",fontSize:14}}>{t.noData}</div>:<>
              {clusters.length>0&&<div style={{marginBottom:20}}>
                <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginBottom:10,textTransform:"uppercase"}}>Event Clusters</div>
                {clusters.map((cl,ci)=><div key={ci} style={{padding:"10px 12px",background:"#0a0a14",borderRadius:8,border:"1px solid #141420",marginBottom:8}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                    <span style={{fontSize:13,color:"#b0b0c8"}}>Type {ci+1}</span>
                    <span style={{fontSize:12,color:"#6a6a8a"}}>{cl.count} events</span></div>
                  <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>{CD.map((d,i)=>
                    <span key={d} style={{fontSize:11,color:`rgba(${CC[d].join(",")},0.8)`,padding:"2px 6px",background:"#0e0e1a",borderRadius:4}}>
                      {d.slice(0,3)} {cl.centroid[i].toFixed(2)}</span>)}</div></div>)}</div>}

              <div style={{fontSize:12,color:"#7a7a9a",letterSpacing:1,marginBottom:10,textTransform:"uppercase"}}>Event Log ({extremeLog.length})</div>
              {extremeLog.slice().reverse().slice(0,20).map((ev,i)=>{
                const d=new Date(ev.time);const ts=`${d.getMonth()+1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2,"0")}`;
                return <div key={i} style={{padding:"8px 12px",background:i%2===0?"#0a0a14":"transparent",borderRadius:6,marginBottom:2}}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:11,color:"#6a6a8a"}}>
                    <span>{ts}</span><span>{ev.deviation.toFixed(1)}σ</span></div>
                  <div style={{display:"flex",gap:4,marginTop:4,flexWrap:"wrap"}}>{CD.map((d,j)=>
                    <span key={d} style={{fontSize:10,color:Math.abs(ev.core[j])>0.3?`rgb(${CC[d].join(",")})`:""+"#5a5a7a",
                      padding:"1px 4px",background:"#0e0e1a",borderRadius:3}}>{d.slice(0,3)} {ev.core[j].toFixed(1)}</span>)}</div></div>;})}
            </>}
          </>}
        </div>
      </div>}

      {/* ═══ CHAT AREA ═══ */}
      <div style={{flex:1,overflowY:"auto",padding:"12px 16px 100px 16px",display:"flex",flexDirection:"column",gap:10,WebkitOverflowScrolling:"touch",minHeight:0}}>
        {!activeChat||!activeChat.messages.length?
          <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:12,opacity:.4}}>
            <div style={{fontSize:40}}>◉</div><div style={{fontSize:16,color:"#7a7a9a"}}>{t.start}</div></div>:
          activeChat.messages.map((msg,i)=><div key={i} style={{maxWidth:"82%",padding:"12px 16px",
            borderRadius:msg.role==="user"?"20px 20px 4px 20px":"20px 20px 20px 4px",
            background:msg.role==="user"?"#1a1a34":"#121220",color:"#e8e8fa",
            alignSelf:msg.role==="user"?"flex-end":"flex-start",fontSize:15,lineHeight:1.6,wordBreak:"break-word",whiteSpace:"pre-wrap"}}>{msg.text}</div>)}
        {loading&&<div style={{maxWidth:"82%",padding:"12px 16px",borderRadius:"20px 20px 20px 4px",background:"#121220",color:"#7a7a9a",alignSelf:"flex-start",fontSize:15}}>
          <span style={{animation:"pulse 1.2s infinite"}}>●●●</span></div>}
        <div ref={chatEnd}/>
      </div>

      {/* INPUT BAR */}
      <div style={{display:"flex",gap:8,padding:"10px 16px 24px 16px",borderTop:"1px solid #141420",background:"#06060c",alignItems:"flex-end",position:"fixed",bottom:0,left:0,right:0,zIndex:30}}>
        <textarea value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMsg();}}}
          placeholder={t.type} rows={2} style={{flex:1,padding:"10px 14px",background:"#10101e",border:"1px solid #1e1e30",borderRadius:16,color:"#e8e8fa",fontSize:16,outline:"none",fontFamily:"inherit",resize:"none",overflow:"auto",minHeight:44,maxHeight:100,lineHeight:1.4}}/>
        {loading?<button onClick={()=>setLoading(false)} style={{padding:"10px 18px",background:"#3a1520",border:"none",borderRadius:16,color:"#ff6b6b",fontSize:15,fontFamily:"inherit",cursor:"pointer",fontWeight:600,flexShrink:0}}>{t.stop}</button>:
          <button onClick={sendMsg} disabled={!input.trim()} style={{padding:"10px 18px",background:"#252548",border:"none",borderRadius:16,color:"#c0c0e0",fontSize:15,fontFamily:"inherit",cursor:"pointer",fontWeight:600,flexShrink:0,opacity:input.trim()?1:.4}}>{t.send}</button>}
      </div>
    </div>
  );
}
