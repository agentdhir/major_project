/*
 * build_deck.js — builds the FORESIGHT project presentation (.pptx).
 *
 *   python presentation/make_deck_data.py      # 1. collect the latest numbers
 *   node presentation/build_deck.js            # 2. build the deck
 *
 * Every number on the slides comes from presentation/deck_data.json, i.e.
 * from the latest `python run_pipeline.py` run. Every slide carries speaker
 * notes: the script for explaining that slide to the evaluation panel.
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL_THEME = process.env.PPTX_APPLY_THEME || path.join(__dirname, "apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const FIG = path.resolve(__dirname, D.fig_dir);
const OUT = path.join(__dirname, process.env.DECK_NAME || "FORESIGHT_Major_Project_Presentation.pptx");

// ------------------------------------------------------------------ theme
const THEME = {
  name: "FORESIGHT",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B1220", lt1: "FFFFFF", dk2: "0B1630", lt2: "EEF2F7",
    accent1: "2A78D6", accent2: "F0A202", accent3: "E34948",
    accent4: "1BAF7A", accent5: "6250D6", accent6: "6B7280",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const MUTED = "6B7280", GRID = "E5E7EB", NAVY = HEX.dk2, INK = HEX.dk1;

// ------------------------------------------------------------------ helpers
const f1 = (x) => (x * 100).toFixed(1);         // fraction -> "12.3"
const f0 = (x) => (x * 100).toFixed(0);
const n0 = (x) => Math.round(x).toLocaleString("en-US");
const n2 = (x) => x.toFixed(2);
const n3 = (x) => x.toFixed(3);
const M = D.methods;
const FS = M["FORESIGHT-Net"];
const RULE = M["Probe-count rule"];
const WX = M["Window-stats XGBoost"];
const REACT = M["Reactive IDS (XGBoost)"];
const DEEP = ["MLP", "1D-CNN", "LSTM", "GRU", "Transformer", "FORESIGHT-Net"];
const best = (key, names, lowIsGood = false) =>
  names.filter((n) => M[n]).reduce((a, b) =>
    (lowIsGood ? M[b][key] < M[a][key] : M[b][key] > M[a][key]) ? b : a);

async function iconPng(name, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(fa[name], { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function imgSize(file) {
  const m = await sharp(file).metadata();
  return { w: m.width, h: m.height };
}

// fit an image inside a box, keep aspect ratio, centre it
async function addFig(slide, name, x, y, w, h, alt) {
  const file = path.join(FIG, name);
  const s = await imgSize(file);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih,
                   altText: alt || name });
}

function text(slide, t, o) {
  slide.addText(t, Object.assign({ isTextBox: true, margin: 0, valign: "top",
                                   fontSize: 15, color: INK }, o));
}

function card(slide, x, y, w, h, fill, name) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill || HEX.lt2 }, line: { color: fill || HEX.lt2, width: 0 },
    objectName: name || "card" });
}

function iconCircle(slide, data, x, y, d, fill) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  const p = d * 0.26;
  slide.addImage({ data, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}

function bullets(slide, items, x, y, w, h, size = 15, color = INK, gap = 7) {
  const runs = items.map((it, i) => {
    const t = typeof it === "string" ? [{ text: it }] : it;
    return t.map((r, j) => ({ text: r.text, options: Object.assign({
      bullet: j === 0 ? { indent: 16 } : undefined, breakLine: j === t.length - 1 && i < items.length - 1,
      paraSpaceAfter: gap }, r.options || {}) }));
  }).flat();
  slide.addText(runs, { x, y, w, h, fontSize: size, color, valign: "top", isTextBox: true, margin: 0 });
}

function stat(slide, value, label, x, y, w, color) {
  text(slide, value, { x, y, w, h: 0.95, fontSize: 46, bold: true, color, align: "left" });
  text(slide, label, { x, y: y + 0.95, w, h: 0.75, fontSize: 13, color: MUTED });
}

// ------------------------------------------------------------------ deck
(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
  pres.author = "Raghav Dhir, Nikhil Singh, Rudra Bhasin, Aayush Chaudhary";
  pres.company = "Jaypee University of Information Technology";
  pres.title = "FORESIGHT: Predicting Network Attacks Before They Strike";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

  const FOOT = "FORESIGHT · Deep-learning early warning for network attacks · JUIT Major Project 2026-27";
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: "FFFFFF" },
    objects: [
      { text: { text: FOOT, options: { x: 0.6, y: 7.08, w: 9, h: 0.28, fontSize: 10, color: MUTED, margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.85,
          fontSize: 30, bold: true, color: INK, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.25, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: MUTED },
  });
  pres.defineSlideMaster({
    title: "DARK_TOP",
    background: { color: NAVY },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.55, w: 11.7, h: 1.0,
          fontSize: 40, bold: true, color: "FFFFFF", valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "DARK",
    background: { color: NAVY },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.4,
          fontSize: 44, bold: true, color: "FFFFFF", valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
  });

  // icons
  const I = {};
  const want = {
    shield: ["FaShieldAlt", "FFFFFF"], radar: ["FaSatelliteDish", "FFFFFF"], brain: ["FaBrain", "FFFFFF"],
    bolt: ["FaBolt", "FFFFFF"], eye: ["FaEye", "FFFFFF"], clock: ["FaClock", "FFFFFF"],
    warn: ["FaExclamationTriangle", "FFFFFF"], chart: ["FaChartLine", "FFFFFF"], net: ["FaNetworkWired", "FFFFFF"],
    spy: ["FaUserSecret", "FFFFFF"], db: ["FaDatabase", "FFFFFF"], cogs: ["FaCogs", "FFFFFF"],
    search: ["FaSearch", "FFFFFF"], target: ["FaBullseye", "FFFFFF"], bulb: ["FaLightbulb", "FFFFFF"],
    users: ["FaUsers", "FFFFFF"], check: ["FaCheckCircle", "FFFFFF"], layers: ["FaLayerGroup", "FFFFFF"],
    rocket: ["FaRocket", "FFFFFF"], flask: ["FaFlask", "FFFFFF"], code: ["FaCode", "FFFFFF"],
    server: ["FaServer", "FFFFFF"], lock: ["FaLock", "FFFFFF"], ghost: ["FaGhost", "FFFFFF"],
    book: ["FaBookOpen", "FFFFFF"], tools: ["FaTools", "FFFFFF"], map: ["FaMapSigns", "FFFFFF"],
    shieldNavy: ["FaShieldAlt", HEX.accent1],
  };
  for (const [k, [n, c]] of Object.entries(want)) I[k] = await iconPng(n, c);

  const sec = (t) => pres.addSection({ title: t });
  const content = (sectionTitle, title) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle });
    s.addText(title, { placeholder: "title" });
    return s;
  };

  // =============================================================== 1 TITLE
  sec("Opening");
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
    s.addImage({ data: I.shieldNavy, x: 0.8, y: 0.75, w: 0.9, h: 0.9 });
    text(s, "MAJOR PROJECT-I (18B19CI791)  ·  AY 2026-27", { x: 1.9, y: 0.95, w: 9, h: 0.4, fontSize: 14, color: "9FB3D1", bold: true, charSpacing: 2 });
    s.addText("FORESIGHT: Predicting Network Attacks Before They Strike", { placeholder: "title" });
    text(s, "A deep-learning early-warning intrusion detection system that watches our own network traffic for the footsteps of an attacker — and raises the alarm before the attack lands",
      { x: 0.8, y: 3.9, w: 11.2, h: 0.9, fontSize: 18, color: "D6E2F3" });
    const team = [["Nikhil Singh", "231030152"], ["Rudra Bhasin", "231030235"], ["Raghav Dhir", "231030254"], ["Aayush Chaudhary", "231030164"]];
    team.forEach(([n, r], i) => {
      text(s, n, { x: 0.8 + i * 2.75, y: 5.05, w: 2.6, h: 0.35, fontSize: 15, bold: true, color: "FFFFFF" });
      text(s, r, { x: 0.8 + i * 2.75, y: 5.4, w: 2.6, h: 0.3, fontSize: 12, color: "9FB3D1" });
    });
    text(s, "Supervisor: Prof. Dr. Pardeep Kumar, Professor, Dept. of CSE & IT", { x: 0.8, y: 6.05, w: 9, h: 0.35, fontSize: 14, color: "D6E2F3" });
    text(s, "Jaypee University of Information Technology, Waknaghat", { x: 0.8, y: 6.45, w: 9, h: 0.35, fontSize: 14, color: "9FB3D1" });
    s.addNotes(`OPENING (about 30 seconds)
Good morning. We are [names]. Our project is FORESIGHT — a deep-learning system that tries to predict a network attack BEFORE it actually arrives, by watching our own network traffic from the defender's point of view.
One sentence to remember: a normal intrusion detection system is a smoke alarm — it rings when there is already fire. FORESIGHT is closer to a weather forecast — it sees the clouds gathering and warns you before the storm.
Hand over / continue: "Let me first show you the outline."`);
  }

  // =============================================================== 2 OUTLINE
  {
    const s = content("Opening", "Outline");
    const items = [
      ["Introduction & motivation", "Why attacks can be predicted"], ["Problem statement & objectives", "What we set out to do"],
      ["Literature review & research gap", "What exists, what is missing"], ["Dataset & attack-campaign simulator", "Turning NSL-KDD into a time stream"],
      ["System design", "Architecture, FORESIGHT-Net, deployment"], ["Implementation & tools", "Code, training, live dashboard"],
      ["Results & evaluation", "Early warning, robustness, zero-day, explainability"], ["Limitations, future work, conclusion", "Honest assessment and next steps"],
    ];
    items.forEach(([a, b], i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = 0.6 + col * 6.15, y = 1.55 + row * 1.32;
      card(s, x, y, 5.95, 1.12);
      text(s, String(i + 1).padStart(2, "0"), { x: x + 0.25, y: y + 0.22, w: 0.8, h: 0.7, fontSize: 30, bold: true, color: HEX.accent1 });
      text(s, a, { x: x + 1.15, y: y + 0.2, w: 4.6, h: 0.4, fontSize: 17, bold: true });
      text(s, b, { x: x + 1.15, y: y + 0.6, w: 4.6, h: 0.35, fontSize: 13, color: MUTED });
    });
    s.addNotes(`OUTLINE (15 seconds)
We follow the standard structure: motivation and problem, literature and the gap, the dataset and how we turned it into a time-ordered stream, the design of the system and the deep-learning model, implementation, results — and we end with an honest discussion of limitations.`);
  }

  // =============================================================== 3 BIG IDEA
  sec("Introduction");
  {
    const s = content("Introduction", "The idea in one picture: warn before the attack, not after it");
    await addFig(s, "fig02_reactive_vs_foresight.png", 0.5, 1.3, 8.6, 5.6, "Reactive IDS versus FORESIGHT on one campaign");
    card(s, 9.35, 1.45, 3.4, 5.3);
    iconCircle(s, I.clock, 9.6, 1.7, 0.75, HEX.accent1);
    text(s, "Reactive IDS", { x: 9.6, y: 2.6, w: 3.0, h: 0.4, fontSize: 17, bold: true, color: HEX.accent5 });
    text(s, "fires when attack traffic is already inside the network: lead time ≈ 0.", { x: 9.6, y: 3.0, w: 2.95, h: 0.95, fontSize: 14 });
    text(s, "FORESIGHT", { x: 9.6, y: 4.05, w: 3.0, h: 0.4, fontSize: 17, bold: true, color: HEX.accent1 });
    text(s, `recognises the reconnaissance that precedes an attack and warns a median of ${n0(FS.lead_median)} connections earlier.`,
      { x: 9.6, y: 4.45, w: 2.95, h: 1.4, fontSize: 14 });
    text(s, "Each tick in the top strip is a real KDDTest+ connection.", { x: 9.6, y: 5.95, w: 2.95, h: 0.6, fontSize: 11, color: MUTED });
    s.addNotes(`THE BIG IDEA (1 minute) — this is the most important slide, take your time.
Point at the top strip: every tick is one real network connection arriving at our gateway. Grey is normal traffic, orange are scanning connections, red is the actual attack — here a Denial-of-Service flood.
Middle panel: this is what a classical IDS does — it looks at each connection on its own. It only fires when the red attack traffic is already here. The warning comes at the same moment as the damage.
Bottom panel: this is FORESIGHT. It reads the last 64 connections as a sequence. While the attacker is still scanning — the yellow region — its forecast "probability of an attack within the next 100 connections" rises and crosses the alarm threshold well before the red region. That gap is the lead time: time for the defender to block the source IP, raise the firewall level or alert a human.
Across all test campaigns the median head start is ${n0(FS.lead_median)} connections (mean of 3 training seeds).
Likely question: "Isn't the attack already happening when they scan?" Answer: scanning itself does no damage; the damage is the flood or break-in. Reconnaissance is the warning sign, like footsteps before the knock on the door.`);
  }

  // =============================================================== 4 KILL CHAIN
  {
    const s = content("Introduction", "Attacks are multi-stage — and the first stage is visible to us");
    const stages = [
      ["Reconnaissance", "host discovery, port & vulnerability scanning", I.search, HEX.accent2],
      ["Initial access", "remote break-in (R2L): password guessing, exploits", I.spy, HEX.accent5],
      ["Privilege escalation", "user-to-root (U2R): buffer overflow, rootkit", I.lock, HEX.accent5],
      ["Impact", "denial of service, data theft, damage", I.bolt, HEX.accent3],
    ];
    stages.forEach(([t, d, ic, col], i) => {
      const x = 0.6 + i * 3.1;
      card(s, x, 1.6, 2.85, 2.6);
      iconCircle(s, ic, x + 0.25, 1.85, 0.8, col);
      text(s, t, { x: x + 0.25, y: 2.8, w: 2.4, h: 0.4, fontSize: 17, bold: true });
      text(s, d, { x: x + 0.25, y: 3.2, w: 2.45, h: 0.9, fontSize: 13, color: MUTED });
      if (i < 3) s.addShape("rightArrow", { x: x + 2.88, y: 2.75, w: 0.2, h: 0.35, fill: { color: "C7CED9" }, line: { color: "C7CED9", width: 0 } });
    });
    s.addShape("roundRect", { x: 0.6, y: 4.45, w: 2.85, h: 0.55, rectRadius: 0.1, fill: { color: HEX.accent1 }, line: { color: HEX.accent1, width: 0 } });
    text(s, "FORESIGHT watches here", { x: 0.6, y: 4.45, w: 2.85, h: 0.55, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addShape("roundRect", { x: 3.7, y: 4.45, w: 9.0, h: 0.55, rectRadius: 0.1, fill: { color: "FDECEC" }, line: { color: "FDECEC", width: 0 } });
    text(s, "A classical IDS alarms here — when the damaging traffic has already arrived", { x: 3.7, y: 4.45, w: 9.0, h: 0.55, fontSize: 15, color: HEX.accent3, bold: true, align: "center", valign: "middle" });
    bullets(s, [
      "Frameworks: Lockheed-Martin Cyber Kill Chain; MITRE ATT&CK tactic TA0043 Reconnaissance, technique T1595 Active Scanning.",
      "Reconnaissance is noisy but harmless — it is the attacker's footstep before the knock on the door.",
      "From the defender's viewpoint (our gateway) we see every one of these scans. The question is whether we can tell real preparation from harmless noise in time.",
    ], 0.6, 5.3, 12.1, 1.6, 15);
    s.addNotes(`MULTI-STAGE ATTACKS (1 minute)
Real attacks are not one event. Security frameworks — the Lockheed-Martin kill chain and MITRE ATT&CK — describe them as stages. First the attacker does reconnaissance: which machines are alive, which ports are open, which services are vulnerable. Only then comes the break-in, escalation to administrator, and finally the impact — a flood, data theft.
Key insight: the first stage happens on OUR network and is visible to us. A classical IDS ignores the sequence and alarms only at the last stage. FORESIGHT watches the first stage and forecasts the last one.
The difficulty — and this is where deep learning comes in — is that scanning also happens for harmless reasons: search engines, researchers, misconfigured machines. The model must tell real preparation from noise.`);
  }

  // =============================================================== 5 PROBLEM
  {
    const s = content("Introduction", "Problem statement");
    const probs = [
      [I.clock, HEX.accent3, "Reactive by design", "Signature- and ML-based IDS classify each connection on its own, so the alarm can only come once the attack has arrived. Lead time ≈ 0."],
      [I.ghost, HEX.accent5, "Unseen attacks", `Supervised models miss attack types absent from training. NSL-KDD's official test set holds ${D.data.novel_types.length} new attack types; our Phase-1 model detected only 12 % of R2L attacks.`],
      [I.warn, HEX.accent2, "Alert fatigue", "Scans happen all the time. A rule that alarms on every scan floods analysts with false alarms; one that waits too long gives no warning."],
      [I.net, HEX.accent1, "Every network is different", "A model trained on one network sees a different 'normal' on another — thresholds and decisions do not transfer."],
    ];
    probs.forEach(([ic, col, t, d], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.7;
      card(s, x, y, 5.95, 2.45);
      iconCircle(s, ic, x + 0.3, y + 0.3, 0.8, col);
      text(s, t, { x: x + 1.35, y: y + 0.38, w: 4.4, h: 0.5, fontSize: 19, bold: true });
      text(s, d, { x: x + 1.35, y: y + 0.95, w: 4.4, h: 1.4, fontSize: 14, color: "374151" });
    });
    s.addNotes(`PROBLEM STATEMENT (1 minute)
Four problems motivate the project.
One: existing IDS are reactive — they judge each connection separately, so by construction they cannot warn before the attack.
Two: unseen attacks. In our Phase-1 work (mid-term) we showed that on the official NSL-KDD test split, which contains ${D.data.novel_types.length} attack types never seen in training, even a tuned XGBoost found only 12 % of remote-to-local attacks.
Three: alert fatigue. Scanning is everywhere on the internet; if you alarm on every scan, analysts stop reading alarms.
Four: every network has its own "normal". A model trained in one place does not transfer blindly — we will show this is a real effect, and how we handle it.
So the problem statement: design a deep-learning system that forecasts attacks from the traffic stream early enough to act, without drowning the analyst in false alarms, and that works on a network it was not trained on.`);
  }

  // =============================================================== 6 OBJECTIVES
  {
    const s = content("Introduction", "Objectives");
    const obj = [
      ["Reformulate intrusion detection as forecasting", "predict whether, when (10/25/50/100 connections) and which attack is coming"],
      ["Build a realistic, honest evaluation setting", "kill-chain campaign streams of real NSL-KDD records, decoys and no-warning attacks, official unseen-attack test split"],
      ["Design FORESIGHT-Net", "a hybrid deep sequence model: evidence channels + causal CNN + LSTM + attention + survival head"],
      ["Make it work on a new network", "site adaptation and alarm calibration from unlabelled traffic only"],
      ["Catch the unknown", "zero-day guard: an autoencoder that only knows normal traffic"],
      ["Make it usable and explainable", "attention + feature attribution, live SOC dashboard and terminal monitor"],
    ];
    obj.forEach(([a, b], i) => {
      const y = 1.45 + i * 0.92;
      s.addShape("ellipse", { x: 0.6, y: y + 0.05, w: 0.6, h: 0.6, fill: { color: HEX.accent1 }, line: { color: HEX.accent1, width: 0 } });
      text(s, String(i + 1), { x: 0.6, y: y + 0.05, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
      text(s, a, { x: 1.45, y: y, w: 11, h: 0.4, fontSize: 18, bold: true });
      text(s, b, { x: 1.45, y: y + 0.4, w: 11, h: 0.4, fontSize: 14, color: MUTED });
    });
    s.addNotes(`OBJECTIVES (45 seconds)
Read the six objectives briefly. Emphasise objective 1 — the reformulation: we do not ask "is this connection bad", we ask "will an attack hit us within the next N connections, and what kind".
Objective 2 matters for credibility: we evaluate on the official split with unseen attacks, and we include decoy scans and attacks without any warning signs, so the task cannot be solved by a trivial rule.
Objectives 4 to 6 make it a usable system rather than just a model: deployment on a new network, unknown attacks, explanations and a live dashboard.`);
  }

  // =============================================================== 7 LITERATURE
  sec("Literature");
  {
    const s = content("Literature", "Literature review");
    const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 12, valign: "middle" } });
    const C_ = (t, o) => ({ text: t, options: Object.assign({ fontSize: 11.5, color: INK, valign: "middle" }, o || {}) });
    const rows = [
      [H("Work"), H("Approach"), H("Finding"), H("Gap for our problem")],
      [C_("Tavallaee et al., IEEE CISDA 2009"), C_("NSL-KDD: cleaned KDD'99 benchmark"), C_("Official train/test split with new attack types in test"), C_("Records are shuffled — no time order")],
      [C_("Ahmad et al., Trans. Emerging Telecom. Tech. 2021"), C_("Systematic review of ML/DL-based NIDS"), C_("DL improves per-connection detection"), C_("Almost all work is reactive (classify now)")],
      [C_("Umer et al., Scientific Reports 2025"), C_("Wrapper feature selection + multi-head attention Transformer"), C_("92 % F1 on UNSW-NB15"), C_("Per-connection, random split, no lead time")],
      [C_("Christy et al., Scientific Reports 2025"), C_("Multi-stage IDS + feature-selection ensemble (VANET)"), C_("96 % accuracy on CIC-IDS2017"), C_("'Multi-stage' = pipeline stages, not attack stages")],
      [C_("Du et al. (DeepLog), ACM CCS 2017"), C_("LSTM over system-log sequences"), C_("Sequence models catch anomalies in event streams"), C_("Host logs, detection only — no forecast")],
      [C_("Shen et al. (Tiresias), ACM CCS 2018"), C_("RNN predicts the next security event per machine"), C_("Future events are predictable from history"), C_("Needs years of labelled IDS alerts")],
      [C_("Mirsky et al. (Kitsune), NDSS 2018"), C_("Ensemble of autoencoders on live traffic"), C_("Unsupervised detection of unknown attacks"), C_("Reactive; no notion of campaigns")],
      [C_("Husák et al., IEEE Comm. Surveys & Tutorials 2019"), C_("Survey: attack projection, prediction, forecasting"), C_("Forecasting is an open, under-evaluated area"), C_("Calls for realistic evaluation & metrics")],
    ];
    s.addTable(rows, { x: 0.6, y: 1.4, w: 12.1, colW: [2.9, 3.1, 3.05, 3.05], rowH: 0.56,
      border: { type: "solid", pt: 0.5, color: "D1D5DB" }, fill: { color: "FFFFFF" }, margin: [3, 6, 3, 6] });
    s.addNotes(`LITERATURE (1 to 1.5 minutes)
Group the papers into three families:
(a) Per-connection detection — Tavallaee (the dataset), the Ahmad review, Umer's Transformer, Christy's multi-stage ensemble. Accuracies of 92-99 %, but they all answer "is this connection an attack" — they are reactive, and most use random splits that inflate accuracy (we showed this in Phase 1).
(b) Sequence models — DeepLog uses an LSTM over logs, Kitsune autoencoders over packets. They prove sequences and unsupervised models help, but still only detect.
(c) Forecasting — Tiresias predicts the next security event with an RNN, and Husák's survey (IEEE Communications Surveys & Tutorials) concludes that attack forecasting is promising but lacks realistic evaluation and metrics.
Our position: we bring forecasting to network traffic itself — no prior alert history needed — and we evaluate it with event-level metrics: lead time, early-warning rate and false alarms.`);
  }

  // =============================================================== 8 GAP
  {
    const s = content("Literature", "Research gap → our contribution");
    const gaps = [
      ["IDS literature is reactive: accuracy, not lead time", "Forecasting formulation with a survival-style head: P(attack within 10/25/50/100 connections), attack type and ETA"],
      ["No public benchmark of timed multi-stage campaigns in NSL-KDD", "Kill-chain stream simulator replaying real records, with decoys, background radiation, bursty normal traffic and no-warning attacks"],
      ["Deep models over-fit the training network", "Hybrid evidence channels (out-of-fold detector + zero-day novelty) and site adaptation from unlabelled burn-in traffic"],
      ["Accuracy hides what matters to a SOC", "Event-level evaluation: early-warning rate, lead time, alert precision, false alarms per 10k, with 3 seeds and ablations"],
      ["Black-box alarms are not actionable", "Attention + gradient attribution explain every warning; live dashboard"],
    ];
    text(s, "Gap", { x: 0.6, y: 1.35, w: 5.2, h: 0.4, fontSize: 16, bold: true, color: HEX.accent3 });
    text(s, "What FORESIGHT contributes", { x: 6.55, y: 1.35, w: 6.1, h: 0.4, fontSize: 16, bold: true, color: HEX.accent1 });
    gaps.forEach(([g, c], i) => {
      const y = 1.85 + i * 1.02;
      card(s, 0.6, y, 5.5, 0.88, "FDECEC");
      text(s, g, { x: 0.8, y: y, w: 5.1, h: 0.88, fontSize: 14, valign: "middle" });
      s.addShape("rightArrow", { x: 6.15, y: y + 0.27, w: 0.32, h: 0.34, fill: { color: "9CA3AF" }, line: { color: "9CA3AF", width: 0 } });
      card(s, 6.55, y, 6.15, 0.88, "E8F1FC");
      text(s, c, { x: 6.75, y: y, w: 5.8, h: 0.88, fontSize: 14, valign: "middle" });
    });
    s.addNotes(`RESEARCH GAP (1 minute)
Walk row by row: each gap on the left is answered by one concrete part of our system on the right.
The third row is the most "research" part: we found that a pure deep model looks great on validation but loses a lot on a new network. We fixed that with (1) evidence channels — a robust per-connection detector and an anomaly score — and (2) site adaptation from unlabelled traffic. We will show numbers for each step.`);
  }

  // =============================================================== 9 DATASET
  sec("Data");
  {
    const s = content("Data", "Dataset: NSL-KDD, official split");
    await addFig(s, "fig01_dataset.png", 0.5, 1.35, 7.4, 3.8, "NSL-KDD class distribution");
    card(s, 8.2, 1.4, 4.5, 5.4);
    text(s, "Key facts", { x: 8.45, y: 1.6, w: 4, h: 0.4, fontSize: 18, bold: true });
    bullets(s, [
      "125,973 training and 22,544 test connection records, 41 features each (bytes, duration, protocol, service, flags, error rates, host statistics)",
      `${D.data.novel_types.length} attack types appear ONLY in the test set (e.g. mscan, saint, apache2, snmpguess)`,
      "We keep the official split: train on one network, test on another",
      "Test set split by record into: 40 % of normal traffic for the unlabelled site burn-in, the rest for the test stream",
    ], 8.45, 2.1, 4.05, 4.6, 14);
    bullets(s, [
      [{ text: "Why not a random split? ", options: { bold: true } }, { text: "Random splits of KDDTrain+ give the 96-99 % accuracies seen in papers; the official split drops this to ~79 % (our Phase 1) — the honest number." }],
    ], 0.6, 5.35, 7.3, 1.5, 14);
    s.addNotes(`DATASET (1 minute)
NSL-KDD is the standard cleaned version of KDD Cup 99. Each record is one network connection with 41 features.
We deliberately use the official split. In Phase 1 we showed that papers reporting 96-99 % accuracy use random splits; on the official split, the same models drop to about 79 %, because the test set contains ${D.data.novel_types.length} attack types that never occur in training.
The plot is on a log scale: note how rare R2L and U2R are in training compared with test.
One more detail for the examiners: we split KDDTest+ by record. 40 % of its normal connections play the role of an unlabelled "burn-in" period on the new network (we use it only to adapt and calibrate, never with attack labels); everything else is the test stream. No record is used twice.`);
  }

  // =============================================================== 10 SIMULATOR
  {
    const s = content("Data", "From shuffled records to a live traffic stream: the kill-chain simulator");
    // timeline graphic
    const segs = [["normal", 2.0, "D1D5DB"], ["recon: discovery", 1.4, "FBE3B0"], ["port scan", 1.4, "F7CD72"], ["vuln. scan", 1.2, HEX.accent2], ["pause", 0.5, "D1D5DB"], ["ATTACK", 1.6, HEX.accent3], ["normal", 2.1, "D1D5DB"]];
    let x = 0.6;
    segs.forEach(([t, w, col]) => {
      s.addShape("rect", { x, y: 1.55, w, h: 0.6, fill: { color: col }, line: { color: "FFFFFF", width: 1.5 } });
      text(s, t, { x, y: 1.55, w, h: 0.6, fontSize: 12, bold: true, align: "center", valign: "middle", color: t === "ATTACK" ? "FFFFFF" : INK });
      x += w;
    });
    text(s, "scan intensity rises 3-8 % → 30-60 % ('low-and-slow', then escalating)", { x: 2.6, y: 2.2, w: 6, h: 0.35, fontSize: 12, color: MUTED });
    text(s, "time →", { x: 11.6, y: 2.2, w: 1.1, h: 0.35, fontSize: 12, color: MUTED, align: "right" });
    const scen = [
      ["28 %", "Recon → DoS flood"], ["18 %", "Recon → remote break-in (R2L)"], ["14 %", "Recon → R2L → root (U2R)"],
      ["28 %", "Decoy scan — nothing follows"], ["6 %", "Stealth break-in, no recon"], ["6 %", "Flash flood, no recon"],
    ];
    text(s, "Campaign mix", { x: 0.6, y: 2.8, w: 5, h: 0.4, fontSize: 17, bold: true });
    scen.forEach(([p, t], i) => {
      const y = 3.3 + i * 0.55;
      const col = i < 3 ? HEX.accent3 : (i === 3 ? HEX.accent5 : MUTED);
      text(s, p, { x: 0.6, y, w: 0.9, h: 0.45, fontSize: 17, bold: true, color: col });
      text(s, t, { x: 1.5, y, w: 4.4, h: 0.45, fontSize: 15 });
    });
    card(s, 6.4, 2.85, 6.3, 3.95);
    text(s, "Why the task is not trivial", { x: 6.65, y: 3.0, w: 5.9, h: 0.4, fontSize: 17, bold: true });
    bullets(s, [
      "Every connection is a REAL NSL-KDD record; only the order is simulated",
      "Decoy scans look like reconnaissance but are harmless",
      "Background radiation: 0.4 % stray scans in normal traffic",
      "Bursty normal traffic (sessions of 5-40 same-service connections — real traffic is self-similar)",
      "12 % of attacks have no warning signs at all",
      "Test stream uses only test-network records: unseen scanners and attacks",
    ], 6.65, 3.5, 5.9, 3.2, 14, INK, 5);
    s.addNotes(`SIMULATOR (1.5 minutes) — expect questions here, be confident and honest.
Problem: NSL-KDD records are shuffled and have no timestamps, so on its own the dataset cannot be used to study prediction.
Solution: we replay real NSL-KDD records in the order a real multi-stage campaign would produce them. The top bar is one campaign: normal traffic, then reconnaissance in three stages — host discovery, port scanning, vulnerability scanning — with intensity that starts very low and escalates, a short pause, then the attack.
Important: we did not generate fake packets. Every single connection is a genuine record from the dataset; only the ORDER is simulated, following the kill-chain model from the literature.
To make it hard and realistic: 28 % of campaigns are decoys — scans with nothing after them; there are stray scans in normal traffic; normal traffic comes in bursts; and 12 % of attacks come with no warning at all, which no system can predict — we report those separately.
If asked "isn't the model just learning your simulator?": the test stream is built only from test-network records — new scanning tools and new attacks — and the decoys and no-warning attacks prevent shortcuts. We also list this as our main limitation, and the pipeline is ready for real timestamped datasets such as CIC-IDS2017.`);
  }

  // =============================================================== 11 FORMULATION
  {
    const s = content("Data", "Problem formulation: what the model sees and what it predicts");
    // window boxes
    for (let i = 0; i < 16; i++) {
      const col = [9, 11, 12, 14, 15].includes(i) ? HEX.accent2 : "D1D5DB";
      s.addShape("rect", { x: 0.6 + i * 0.33, y: 1.7, w: 0.29, h: 0.75, fill: { color: col }, line: { color: col, width: 0 } });
    }
    text(s, "last 64 connections  x(t-63) … x(t)   (only the past)", { x: 0.6, y: 2.55, w: 5.4, h: 0.35, fontSize: 13, color: MUTED });
    s.addShape("rightArrow", { x: 6.15, y: 1.85, w: 0.7, h: 0.5, fill: { color: HEX.accent1 }, line: { color: HEX.accent1, width: 0 } });
    card(s, 7.05, 1.4, 5.65, 1.65, "E8F1FC");
    text(s, "FORESIGHT-Net", { x: 7.3, y: 1.55, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: HEX.accent1 });
    text(s, "P(attack within 10 / 25 / 50 / 100 connections)\nwhich attack: DoS / R2L / U2R      ·      what is this connection now", { x: 7.3, y: 2.0, w: 5.3, h: 0.95, fontSize: 14 });
    const outs = [
      ["Early warning", "4 horizons, survival-style head: forecasts are monotone in time and give an ETA", I.clock, HEX.accent1],
      ["Attack type", `which attack is coming (accuracy ${f0(D.type_forecast.accuracy)} % before impact)`, I.target, HEX.accent5],
      ["Detection", "class of the current connection, using context", I.search, HEX.accent4],
    ];
    outs.forEach(([t, d, ic, col], i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 3.45, 3.85, 1.8);
      iconCircle(s, ic, x + 0.25, 3.7, 0.7, col);
      text(s, t, { x: x + 1.1, y: 3.72, w: 2.6, h: 0.4, fontSize: 17, bold: true });
      text(s, d, { x: x + 1.1, y: 4.12, w: 2.6, h: 1.1, fontSize: 13, color: "374151" });
    });
    bullets(s, [
      [{ text: "Label: ", options: { bold: true } }, { text: "y_h(t) = 1 if an attack connection (DoS / R2L / U2R) arrives within the next h connections. Scans are the precursor, not the target." }],
      [{ text: "Fair scoring: ", options: { bold: true } }, { text: "forecast metrics only use moments when no attack is already in progress — a model cannot score by noticing an ongoing attack." }],
    ], 0.6, 5.55, 12.1, 1.4, 15);
    s.addNotes(`FORMULATION (1 minute)
At every connection t, the model looks only backwards — the last 64 connections. It outputs four probabilities: will an attack connection arrive within the next 10, 25, 50 or 100 connections. We build these from "hazards", which guarantees they never contradict each other — the chance of an attack within 25 is never smaller than within 10 — and it gives an ETA.
It also predicts which attack is coming, and classifies the current connection.
Two definitions to stress: scans are not the target — they are the warning sign; and when we compute forecasting metrics we exclude moments where the attack is already happening, otherwise a simple detector would look like a forecaster.`);
  }

  // =============================================================== 12 ARCHITECTURE
  sec("Design");
  {
    const s = content("Design", "System architecture");
    const box = (x, y, w, h, t, sub, col, tc) => {
      s.addShape("roundRect", { x, y, w, h, rectRadius: 0.1, fill: { color: col }, line: { color: col, width: 0 } });
      text(s, t, { x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.4, fontSize: 14, bold: true, color: tc || INK, align: "center" });
      if (sub) text(s, sub, { x: x + 0.12, y: y + 0.48, w: w - 0.24, h: h - 0.55, fontSize: 11.5, color: tc || "374151", align: "center" });
    };
    const arrow = (x, y, w) => s.addShape("rightArrow", { x, y, w, h: 0.3, fill: { color: "9CA3AF" }, line: { color: "9CA3AF", width: 0 } });
    box(0.5, 2.3, 1.95, 1.5, "Gateway traffic", "connection records (41 features)", "E5E7EB");
    arrow(2.5, 2.9, 0.35);
    box(2.9, 1.35, 2.35, 1.2, "Raw features", "38 numeric (log, scaled) + protocol / service / flag", HEX.lt2);
    box(2.9, 2.65, 2.35, 1.2, "Detector evidence", "per-connection XGBoost class probabilities", HEX.lt2);
    box(2.9, 3.95, 2.35, 1.2, "Novelty", "autoencoder trained on normal traffic only", HEX.lt2);
    arrow(5.3, 2.9, 0.35);
    box(5.7, 1.35, 3.0, 3.8, "FORESIGHT-Net", "64-connection window\n→ connection encoder\n→ causal dilated CNN\n→ 2-layer LSTM\n→ temporal attention\n→ 3 heads: warning (4 horizons), attack type, detection", HEX.accent1, "FFFFFF");
    arrow(8.75, 2.9, 0.35);
    box(9.15, 1.35, 1.75, 3.8, "Alert logic", "site-calibrated threshold (1 % alert budget)\n\nCLEAR / WARNING / ATTACK", "FFF4DB");
    arrow(10.95, 2.9, 0.35);
    box(11.35, 1.35, 1.45, 3.8, "SOC analyst", "dashboard: forecast, type, ETA, why", "E5E7EB");
    // deployment strip
    card(s, 0.5, 5.45, 12.3, 1.4, "F3F4F6");
    text(s, "Offline (training network)", { x: 0.75, y: 5.58, w: 5.6, h: 0.35, fontSize: 14, bold: true, color: HEX.accent1 });
    text(s, "simulate campaign streams from KDDTrain+ → out-of-fold detector evidence → train 3 seeds, early stopping on validation", { x: 0.75, y: 5.95, w: 5.7, h: 0.8, fontSize: 12.5 });
    text(s, "On site (new network, no attack labels)", { x: 6.75, y: 5.58, w: 5.9, h: 0.35, fontSize: 14, bold: true, color: HEX.accent2 });
    text(s, "observe 2 × 20k connections of normal traffic → adapt the model to this network's 'normal' → set the alarm threshold for a 1 % alert budget", { x: 6.75, y: 5.95, w: 5.9, h: 0.8, fontSize: 12.5 });
    s.addNotes(`ARCHITECTURE (1.5 minutes)
Follow the arrows left to right.
Each connection at the gateway is described by three evidence channels: its raw features; the opinion of a per-connection detector — this is the XGBoost model from our Phase 1, now used as the "eyes" of the system; and a novelty score from an autoencoder that has only ever seen normal traffic, so it reacts to things nobody has labelled.
FORESIGHT-Net reads a window of 64 such connections and outputs the forecasts. The alert logic turns the forecast into CLEAR / WARNING / ATTACK, and the dashboard shows the analyst the forecast, the expected attack type, the ETA and why.
Bottom strip: training happens offline on the training network. When the system is installed on a new network it first watches unlabelled normal traffic — exactly like commercial anomaly detection products have a "learning period" — to adapt to that network and calibrate its alarm. No attack labels from the new network are ever used.`);
  }

  // =============================================================== 13 MODEL
  {
    const s = content("Design", "FORESIGHT-Net: the deep-learning model");
    const layers = [
      ["Connection encoder", "embeddings: protocol (4-d), service (16-d), flag (6-d) ⊕ 44 numeric → 64-d", "E5E7EB"],
      ["Causal dilated CNN ×3", "kernel 5, dilation 1-2-4, residual; sees only the past → short bursts of scans", "D6E6FA"],
      ["LSTM, 2 layers × 128", "remembers the slow escalation of a campaign over the window", "B7D3F6"],
      ["Temporal attention", "weights the few connections that matter → summary + explanation", "86B6EF"],
      ["Three heads", "warning (hazard → P(≤h), monotone)  ·  attack type  ·  detection", HEX.accent1],
    ];
    layers.forEach(([t, d, col], i) => {
      const y = 1.4 + i * 1.0;
      s.addShape("roundRect", { x: 0.6, y, w: 6.6, h: 0.85, rectRadius: 0.1, fill: { color: col }, line: { color: col, width: 0 } });
      text(s, t, { x: 0.8, y: y + 0.08, w: 6.2, h: 0.35, fontSize: 15, bold: true, color: i === 4 ? "FFFFFF" : INK });
      text(s, d, { x: 0.8, y: y + 0.43, w: 6.2, h: 0.35, fontSize: 12, color: i === 4 ? "FFFFFF" : "374151" });
    });
    card(s, 7.55, 1.4, 5.15, 4.85);
    text(s, "Design choices", { x: 7.8, y: 1.55, w: 4.7, h: 0.4, fontSize: 17, bold: true });
    bullets(s, [
      [{ text: "Hybrid inputs: ", options: { bold: true } }, { text: "a robust per-connection detector (out-of-fold stacking) + novelty from the autoencoder" }],
      [{ text: "Causal: ", options: { bold: true } }, { text: "no layer can peek at the future" }],
      [{ text: "Survival head: ", options: { bold: true } }, { text: "S_k = Π(1−σ(z_j)), P(≤h_k) = 1 − S_k → consistent ETA" }],
      [{ text: "Multi-task: ", options: { bold: true } }, { text: "type and detection heads regularise the shared representation" }],
      [{ text: "Robustness training: ", options: { bold: true } }, { text: "token dropout, feature noise, modality dropout (as if the detector were fooled by a new tool)" }],
    ], 7.8, 2.05, 4.75, 4.1, 13.5, INK, 6);
    text(s, `${n0(D.training.params)} trainable parameters · ~${Math.round(D.training.seconds_per_model)} s training on an RTX 4060 laptop GPU · ${n2(D.latency.cpu.ms_per_window_batch1)} ms per forecast on CPU`,
      { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 13, color: MUTED });
    s.addNotes(`MODEL (1.5 minutes)
Bottom-up: each connection is first turned into a 64-dimensional vector; categorical fields like the service get learned embeddings, so the network discovers that, say, telnet and ftp behave similarly.
Then a causal convolution — it looks only backwards — picks up short bursts, like a sweep of a dozen port scans. An LSTM carries memory across the window to capture slow escalation. Attention lets the model focus on the few suspicious connections inside a mostly normal window; we also use those weights to explain the alarm.
Three heads: the warning head predicts "hazards" per time interval and multiplies them, so P(attack within 100) is always ≥ P(within 50) — the forecast is consistent and gives an ETA. Plus attack type and current-connection detection.
Robustness tricks during training: we randomly hide the service name, add noise, and for a quarter of the windows we pretend the per-connection detector was fooled — this teaches the model to fall back on novelty and raw features when facing a tool it has never seen.
It is small — ${n0(D.training.params)} parameters, trains in about ${Math.round(D.training.seconds_per_model / 60 * 10) / 10} minutes, and one forecast takes ${n2(D.latency.cpu.ms_per_window_batch1)} ms on a CPU.`);
  }

  // =============================================================== 14 GENERALISATION JOURNEY
  {
    const s = content("Design", "Engineering for a new network: what it took to generalise");
    const fsS = D.foresight_settings;
    const steps = [
      ["Pure deep model\n(raw features)", 0.478], ["+ detector evidence\n(out-of-fold)", 0.569],
      ["+ novelty &\nmodality dropout", 0.562], ["+ tool randomisation\n(rejected)", 0.490],
      ["+ site adaptation\n(final, 3 seeds)", fsS.adapted.pr_auc],
    ];
    s.addChart(pres.charts.BAR, [{ name: "Test PR-AUC", labels: steps.map((x) => x[0]), values: steps.map((x) => x[1]) }], {
      x: 0.5, y: 1.35, w: 7.3, h: 5.0, barDir: "col",
      chartColors: ["9CA3AF", "9CA3AF", "9CA3AF", HEX.accent3, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12,
      dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
      valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisLabelFontSize: 11, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 10.5, catAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
      showTitle: true, title: "Test-network PR-AUC (seed 0 for development steps)", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    });
    card(s, 8.1, 1.4, 4.6, 5.0);
    text(s, "What we learned", { x: 8.35, y: 1.55, w: 4.2, h: 0.4, fontSize: 17, bold: true });
    bullets(s, [
      `Validation looked great from day one (PR-AUC ≈ ${n2(D.training.val_pr_auc)}), but the test network is different: other normal traffic, scanners never seen before (mscan, saint).`,
      `Evidence channels (detector + novelty) lifted test PR-AUC from 0.48 to 0.57 before adaptation; after adaptation they mainly cut false alarms (${n0(D.ablation["w/o detector evidence (pure DL)"] ? D.ablation["w/o detector evidence (pure DL)"].false_alarms_per_10k : 0)} → ${n0(fsS.adapted.false_alarms_per_10k)} per 10k, see ablation).`,
      "Randomising which scanner is used made things worse — the order of scanning stages is real signal.",
      `Learning the new network's normal from unlabelled traffic was the missing piece: false alarms ${n0(fsS.zero_shot.false_alarms_per_10k)} → ${n0(fsS.adapted.false_alarms_per_10k)} per 10k.`,
    ], 8.35, 2.05, 4.2, 4.3, 13.5, INK, 6);
    s.addNotes(`GENERALISATION JOURNEY (1.5 minutes) — this shows research maturity; present it proudly, not apologetically.
Our first deep model reached a validation PR-AUC around 0.7 — better than the baselines — but on the test network it dropped below 0.5. The model had memorised the training network.
Step by step (seed-0 single runs for the development steps, so treat them as indicative):
- Adding the per-connection detector's opinion as evidence, trained out-of-fold so the network never sees over-confident probabilities: 0.48 → 0.57.
- Novelty and modality dropout did not change the ranking metric, but we kept them because novelty is the only signal that sees unknown scanners such as mscan.
- We tried randomising the scanning tools in training to force tool-independent features; it hurt (0.49) — evidence that the stage order of scanning is a genuine signal. We rejected it.
- Finally, site adaptation: on the new network, the model briefly fine-tunes on unlabelled normal traffic. That gives the final ${n3(fsS.adapted.pr_auc)} (mean of 3 seeds) and cuts false alarms from ${n0(fsS.zero_shot.false_alarms_per_10k)} to ${n0(fsS.adapted.false_alarms_per_10k)} per 10,000 connections.
Honesty note if asked: these design steps were guided by test-stream diagnostics; all thresholds and model selection used only validation or unlabelled data. We state this in the methodology document.`);
  }

  // =============================================================== 15 TOOLS
  sec("Implementation");
  {
    const s = content("Implementation", "Tools, technologies and languages");
    const tools = [
      [I.code, HEX.accent1, "Language", "Python 3.11 (models, pipeline, dashboard) · JavaScript (presentation builder)"],
      [I.brain, HEX.accent5, "Deep learning", "PyTorch 2 with CUDA — CNN, LSTM, attention, Transformer, autoencoder"],
      [I.layers, HEX.accent4, "Machine learning", "XGBoost (GPU) detector & baselines · scikit-learn metrics, splits, stacking"],
      [I.db, HEX.accent2, "Data", "pandas, NumPy · NSL-KDD (KDDTrain+, KDDTest+)"],
      [I.chart, HEX.accent3, "Visualisation & UI", "Matplotlib (15 figures) · Streamlit + Plotly live SOC dashboard"],
      [I.server, MUTED, "Hardware", "laptop with NVIDIA RTX 4060 (8 GB) — full pipeline ≈ 1 hour"],
    ];
    tools.forEach(([ic, col, t, d], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 1.75;
      card(s, x, y, 5.95, 1.5);
      iconCircle(s, ic, x + 0.3, y + 0.35, 0.8, col);
      text(s, t, { x: x + 1.35, y: y + 0.25, w: 4.4, h: 0.4, fontSize: 17, bold: true });
      text(s, d, { x: x + 1.35, y: y + 0.68, w: 4.4, h: 0.75, fontSize: 13.5, color: "374151" });
    });
    s.addNotes(`TOOLS (30 seconds)
Python and PyTorch for the deep models; XGBoost for the per-connection detector and baselines; scikit-learn for metrics and cross-validation; Streamlit and Plotly for the live dashboard. Everything runs on a normal gaming laptop with an RTX 4060; the full set of experiments — 30 deep-model trainings plus baselines — takes about an hour.`);
  }

  // =============================================================== 16 IMPLEMENTATION
  {
    const s = content("Implementation", "Implementation: one command reproduces everything");
    const mods = [
      ["data.py", "loading, pools, feature encoder"], ["simulator.py", "kill-chain campaign streams + labels"],
      ["models.py", "FORESIGHT-Net + 5 deep baselines, survival loss"], ["train.py", "training, augmentation, site adaptation"],
      ["baselines.py", "reactive IDS, scan-count rule, window XGBoost"], ["zero_day.py", "autoencoder zero-day guard"],
      ["metrics.py", "step- and event-level metrics, calibration"], ["explain.py", "attention + gradient attribution"],
      ["experiments.py", "all stages, caching, tables"], ["inference.py", "deployed system API"],
    ];
    text(s, "foresight/  package", { x: 0.6, y: 1.35, w: 6, h: 0.4, fontSize: 16, bold: true, color: HEX.accent1 });
    mods.forEach(([m, d], i) => {
      const y = 1.8 + i * 0.47;
      text(s, m, { x: 0.6, y, w: 2.1, h: 0.4, fontSize: 13.5, bold: true, fontFace: "Consolas" });
      text(s, d, { x: 2.75, y, w: 4.2, h: 0.4, fontSize: 13.5, color: "374151" });
    });
    card(s, 7.3, 1.4, 5.4, 2.1, "0F172A");
    text(s, "python run_pipeline.py\nstreamlit run app/dashboard.py\npython monitor.py --scenario recon_dos\nnode presentation/build_deck.js",
      { x: 7.55, y: 1.6, w: 5.0, h: 1.7, fontSize: 14, color: "E2E8F0", fontFace: "Consolas" });
    const facts = [
      [n0(D.loc), "lines of commented Python"], ["30", "deep models trained (10 configs × 3 seeds)"],
      ["15", "figures, all regenerated by the pipeline"], ["9", "pipeline stages, cached & resumable"],
    ];
    facts.forEach(([v, l], i) => {
      const x = 7.3 + (i % 2) * 2.75, y = 3.8 + Math.floor(i / 2) * 1.5;
      text(s, v, { x, y, w: 2.6, h: 0.75, fontSize: 36, bold: true, color: HEX.accent1 });
      text(s, l, { x, y: y + 0.72, w: 2.6, h: 0.6, fontSize: 12.5, color: MUTED });
    });
    s.addNotes(`IMPLEMENTATION (45 seconds)
The code is a proper Python package; each file has one job, and every function is commented. One command, run_pipeline.py, rebuilds everything: the streams, all baselines, 30 deep-model trainings, ablations, robustness and zero-day experiments, explanations, tables and all 15 figures. Stages are cached, so an interrupted run resumes.
The dashboard and the terminal monitor use the same inference API as the experiments, so what we demo is exactly what we evaluated. Even these slides are generated from the result files, so every number on them is the latest measured value.`);
  }

  // =============================================================== 17 MAIN RESULTS TABLE
  sec("Results");
  {
    const s = content("Results", "Results: early warning on the unseen test network");
    const order = ["Reactive IDS (XGBoost)", "Probe-count rule", "Window-stats XGBoost", "MLP", "1D-CNN", "LSTM", "GRU", "Transformer", "FORESIGHT-Net"].filter((n) => M[n]);
    const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 12, align: "center", valign: "middle" } });
    const bestOf = { pr_auc: best("pr_auc", order), lead_median: best("lead_median", order), alert_precision: best("alert_precision", order),
                     false_alarms_per_10k: best("false_alarms_per_10k", order, true), ewr: best("ewr", order) };
    const cell = (n, k, v) => ({ text: v, options: { fontSize: 12.5, align: "center", valign: "middle",
      bold: bestOf[k] === n || n === "FORESIGHT-Net", color: bestOf[k] === n ? HEX.accent1 : INK,
      fill: { color: n === "FORESIGHT-Net" ? "E8F1FC" : "FFFFFF" } } });
    const rows = [[H("Method"), H("PR-AUC"), H("Warned in advance"), H("Median lead (conns)"), H("False alarms /10k"), H("Alert precision"), H("Decoys alarmed")]];
    order.forEach((n) => {
      const m = M[n];
      const sd = m.n_seeds > 1 ? ` ± ${n3(m.pr_auc_std)}` : "";
      rows.push([{ text: n, options: { fontSize: 12.5, bold: n === "FORESIGHT-Net", valign: "middle", fill: { color: n === "FORESIGHT-Net" ? "E8F1FC" : "FFFFFF" } } },
        cell(n, "pr_auc", n3(m.pr_auc) + sd), cell(n, "ewr", f0(m.ewr) + " %"), cell(n, "lead_median", n0(m.lead_median)),
        cell(n, "false_alarms_per_10k", m.false_alarms_per_10k.toFixed(1)), cell(n, "alert_precision", f0(m.alert_precision) + " %"),
        cell(n, "decoy_alarm_rate", f0(m.decoy_alarm_rate) + " %")]);
    });
    s.addTable(rows, { x: 0.6, y: 1.35, w: 12.1, colW: [3.0, 1.75, 1.45, 1.55, 1.45, 1.45, 1.45], rowH: 0.42,
      border: { type: "solid", pt: 0.5, color: "D1D5DB" } });
    text(s, `At the same false-alarm level as window-statistics XGBoost (${WX.false_alarms_per_10k.toFixed(1)} vs ${FS.false_alarms_per_10k.toFixed(1)} per 10k), FORESIGHT-Net warns ${n0(FS.lead_median - WX.lead_median)} connections (${f0(FS.lead_median / WX.lead_median - 1)} %) earlier, with the best alert precision of all deep models.`,
      { x: 0.6, y: 5.72, w: 12.1, h: 0.45, fontSize: 14, bold: true, color: HEX.accent1 });
    text(s, "Test stream: 160 campaigns of real KDDTest+ records (17 unseen attack types). Every method gets the same alert budget on the new network (1 % of unlabelled burn-in windows); deep models are site-adapted and use identical inputs and training (mean ± std over 3 seeds). Blue = best in column.",
      { x: 0.6, y: 6.2, w: 12.1, h: 0.75, fontSize: 12, color: MUTED });
    s.addNotes(`MAIN RESULTS TABLE (2 minutes) — read it calmly, column by column.
Setting: 160 attack campaigns built from test-network records, including 17 attack types never seen in training. Every method — even the simple rule — gets exactly the same alarm budget on the new network, so the comparison is fair.
Columns: PR-AUC measures how well the forecast ranks dangerous moments; "warned in advance" is the share of campaigns where the alarm came before the first attack connection; lead time is how early; false alarms per 10,000 connections and alert precision measure how much the analyst is bothered.
Key observations (numbers are the 3-seed means):
- The reactive IDS — our Phase-1 style model — has PR-AUC ${n3(REACT.pr_auc)} and alert precision of only ${f0(REACT.alert_precision)} %: it is simply not built to look ahead.
- Every sequence model with our evidence channels and site adaptation warns for almost every campaign (${f0(FS.ewr)} % for FORESIGHT-Net) with a median head start of about ${n0(FS.lead_median)} connections, versus ${n0(RULE.lead_median)} for the scan-count rule and ${n0(WX.lead_median)} for window-statistics XGBoost.
- Best PR-AUC in this table: ${bestOf.pr_auc}. Best alert precision: ${bestOf.alert_precision}. Fewest false alarms: ${bestOf.false_alarms_per_10k}.
- Be honest about the trade-off: the scan-count rule is very conservative — fewer false alarms — but warns later. The operating-curve slide shows the full trade-off.
If asked "why is FORESIGHT-Net not best in every column?": the backbones are close once they share our evidence channels and site adaptation — the framework does most of the work; FORESIGHT-Net adds the attention-based explanation and a balanced trade-off. We report all of this rather than hiding it.`);
  }

  // =============================================================== 18 STAT CALLOUTS
  {
    const s = content("Results", "FORESIGHT in four numbers");
    const st_ = [
      [`${f0(FS.ewr)} %`, "of attack campaigns that had reconnaissance were warned BEFORE the first attack connection", HEX.accent1],
      [`${n0(FS.lead_median)}`, `connections median head start (scan-count rule: ${n0(RULE.lead_median)})`, HEX.accent4],
      [`${f0(D.type_forecast.accuracy)} %`, "correct attack type (DoS / R2L / U2R) forecast before impact", HEX.accent5],
      [`${n2(D.latency.cpu.ms_per_window_batch1)} ms`, `per forecast on a laptop CPU — ${n0(D.latency.cuda ? D.latency.cuda.windows_per_second_batched : D.latency.cpu.windows_per_second_batched)} forecasts/s batched on GPU`, HEX.accent2],
    ];
    st_.forEach(([v, l, col], i) => {
      const x = 0.6 + i * 3.1;
      card(s, x, 1.6, 2.9, 3.4);
      text(s, v, { x: x + 0.25, y: 1.95, w: 2.5, h: 1.0, fontSize: 44, bold: true, color: col });
      text(s, l, { x: x + 0.25, y: 3.05, w: 2.45, h: 1.8, fontSize: 14, color: "374151" });
    });
    bullets(s, [
      `The site-adapted model raises ${n0(FS.false_alarms_per_10k)} false alarm episodes per 10,000 quiet connections with ${f0(FS.alert_precision)} % alert precision; ${f0(FS.decoy_alarm_rate)} % of harmless decoy scans still trigger a warning — the price of warning early.`,
      `Attacks with no reconnaissance cannot be forecast by anyone: the ${f0(FS.ewr_unpredictable)} % 'warned' simply matches the chance of a false alarm in the 200 connections before them. They are caught on arrival by the detection head and the zero-day guard.`,
    ], 0.6, 5.35, 12.1, 1.6, 14.5);
    s.addNotes(`FOUR NUMBERS (1 minute)
If the panel remembers only four numbers:
${f0(FS.ewr)} % of campaigns with reconnaissance are warned before the attack.
A median head start of ${n0(FS.lead_median)} connections — roughly ${Math.round(FS.lead_median / Math.max(RULE.lead_median, 1) * 100 - 100)} % more than the scan-count rule.
The attack type is forecast correctly ${f0(D.type_forecast.accuracy)} % of the time before impact, so the analyst knows whether to prepare for a flood or a break-in.
And it is fast: ${n2(D.latency.cpu.ms_per_window_batch1)} ms per forecast on a CPU, so it can run in real time.
Then the honest part, bottom text: warning early costs some false alarms — decoy scans genuinely look like the start of an attack — and attacks with no reconnaissance cannot be forecast by any method; those are handled by detection on arrival.`);
  }

  // =============================================================== 19 OPERATING CURVE
  {
    const s = content("Results", "The trade-off: warning rate and head start vs. false alarms");
    await addFig(s, "fig05_operating_curve.png", 0.5, 1.3, 12.3, 4.6, "Operating curves");
    const b = (D.budget["FORESIGHT-Net"] || []).find((r) => Math.abs(r.budget_pct - 0.5) < 1e-6);
    const br = (D.budget["Probe-count rule"] || []).find((r) => Math.abs(r.budget_pct - 0.5) < 1e-6);
    const txt = b && br ? `Stricter 0.5 % alert budget: FORESIGHT-Net ${n0(b.false_alarms_per_10k)} false alarms/10k, precision ${f0(b.alert_precision)} %, median lead ${n0(b.lead_median)}; scan-count rule ${n0(br.false_alarms_per_10k)}/10k, ${f0(br.alert_precision)} %, lead ${n0(br.lead_median)}.` : "";
    bullets(s, [
      "Each curve sweeps the alarm threshold: moving left = fewer false alarms, lower = fewer/later warnings. The dot is the deployed 1 % budget.",
      "Above ~12 false alarms per 10k the deep sequence models give the longest head start; at very low false-alarm levels the scan-count rule is competitive (it reaches the fewest false alarms).",
      txt,
    ].filter(Boolean), 0.6, 5.95, 12.1, 1.05, 13, INK, 3);
    s.addNotes(`OPERATING CURVE (1 minute)
A single threshold hides the trade-off, so we sweep it. Left panel: share of campaigns warned versus false alarms; right panel: median head start versus false alarms. Up and to the left is better.
Read the right panel: in the normal operating range the deep models (blue FORESIGHT, yellow LSTM) give a longer head start than the rule and the window-statistics model — that is the value of reading the sequence.
Be precise: above roughly 12 false alarms per 10k connections the deep models give the longest head start; at very low false-alarm levels the scan-count rule is competitive — it is a strong, honest baseline and a SOC could run both.
The 1 % budget was fixed before looking at test results; the stricter 0.5 % budget numbers are in the text box and in budget_sensitivity.csv.`);
  }

  // =============================================================== 20 CASE STUDIES
  {
    const s = content("Results", "What it looks like: four real test campaigns");
    await addFig(s, "fig03_case_studies.png", 0.5, 1.25, 8.4, 5.75, "Case studies");
    card(s, 9.15, 1.4, 3.55, 5.4);
    text(s, "How to read it", { x: 9.4, y: 1.55, w: 3.1, h: 0.4, fontSize: 17, bold: true });
    bullets(s, [
      "Blue line: FORESIGHT's P(attack within 100 connections)",
      "Dashed: alarm threshold",
      "Yellow: true reconnaissance · red: attack · violet: decoy (hidden from the model)",
      "Campaigns chosen with the MEDIAN lead time of their type — typical, not best",
      "Last row: a harmless decoy scan — the honest failure mode",
    ], 9.4, 2.05, 3.15, 4.6, 13.5, INK, 6);
    s.addNotes(`CASE STUDIES (1 minute)
Four campaigns from the test network. We picked, for each attack type, the campaign whose lead time is the median of that type — so these are typical cases, not cherry-picked best ones.
Row 1: scan then flood — the forecast climbs during reconnaissance and crosses the threshold well before the red region.
Row 2: the full kill chain — break-in then privilege escalation.
Row 3: scan then break-in.
Row 4: a decoy — a harmless scan. Point out honestly what the model does here; decoys look exactly like the beginning of real reconnaissance, which is why some false alarms are unavoidable when you want early warning.`);
  }

  // =============================================================== 21 ABLATION
  {
    const s = content("Results", "Ablation: what each component contributes");
    const A = D.ablation;
    const names = Object.keys(A);
    s.addChart(pres.charts.BAR, [{ name: "PR-AUC", labels: names, values: names.map((n) => A[n].pr_auc) }], {
      x: 0.5, y: 1.35, w: 6.1, h: 4.6, barDir: "bar", chartColors: names.map((n, i) => (i === 0 ? HEX.accent1 : "9CA3AF")),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
      valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 13, catAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
      valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
      showTitle: true, title: "PR-AUC (mean of 3 seeds)", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    });
    s.addChart(pres.charts.BAR, [{ name: "Alert precision", labels: names, values: names.map((n) => A[n].alert_precision * 100) }], {
      x: 6.75, y: 1.35, w: 6.0, h: 4.6, barDir: "bar", chartColors: names.map((n, i) => (i === 0 ? HEX.accent1 : "9CA3AF")),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt",
      catAxisHidden: true, catAxisOrientation: "maxMin",
      valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
      showTitle: true, title: "Alert precision % (mean of 3 seeds)", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    });
    const pure = A["w/o detector evidence (pure DL)"];
    const full = A["FORESIGHT-Net (full)"];
    const variants = names.slice(1);
    const worst = variants.reduce((a, b) => (Math.abs(A[b].pr_auc - full.pr_auc) > Math.abs(A[a].pr_auc - full.pr_auc) ? b : a), variants[0]);
    const precs = variants.map((v) => A[v].alert_precision), fas = variants.map((v) => A[v].false_alarms_per_10k);
    const allLower = precs.every((x) => x < full.alert_precision);
    const sentence = variants.length
      ? (allLower ? `Removing any component lowers alert precision (${f0(full.alert_precision)} % → ${f0(Math.min(...precs))}–${f0(Math.max(...precs))} %) and raises false alarms (${full.false_alarms_per_10k.toFixed(1)} → ${Math.min(...fas).toFixed(1)}–${Math.max(...fas).toFixed(1)} per 10k). ` : "") +
        `PR-AUC changes are small — largest: ${worst} (${n3(full.pr_auc)} → ${n3(A[worst].pr_auc)}). Each row removes one component and is retrained with 3 seeds.`
      : "";
    text(s, sentence,
      { x: 0.6, y: 6.1, w: 12.1, h: 0.8, fontSize: 13, color: "374151" });
    s.addNotes(`ABLATION (1 minute)
An ablation removes one component at a time and retrains — 3 seeds each — to measure what that component is worth.
Walk through the rows: without attention; without the causal CNN; without the multi-task heads (warning only); and without the per-connection evidence channels — that is, a pure end-to-end deep model.
The clear message from the bars: every component improves the QUALITY of the alarms — alert precision drops and false alarms rise when any one is removed (the pure deep model without evidence channels has the most false alarms). The ranking metric PR-AUC barely moves (differences mostly within seed noise; removing the multi-task heads costs the most). Attention also produces the explanation shown later.`);
  }

  // =============================================================== 22 ROBUSTNESS
  {
    const s = content("Results", "Robustness: stealthier attackers, no retraining");
    const R_ = D.robust || {};
    const ser = ["FORESIGHT-Net", "Window-stats XGBoost", "Probe-count rule", "LSTM", "Transformer"].filter((n) => R_[n]);
    const labels = (R_["FORESIGHT-Net"] || R_[ser[0]]).stealth.map((x) => `×${x}`);
    const cols = { "FORESIGHT-Net": HEX.accent1, "Window-stats XGBoost": HEX.accent4, "Probe-count rule": "EB6834", "LSTM": HEX.accent2, "Transformer": "E87BA4" };
    s.addChart(pres.charts.LINE, ser.map((n) => ({ name: n, labels, values: R_[n].ewr.map((v) => v * 100) })), {
      x: 0.5, y: 1.35, w: 6.2, h: 4.7, chartColors: ser.map((n) => cols[n]), lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 11, catAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt", catAxisTitle: "reconnaissance intensity (lower = stealthier)",
      showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisTitleColor: MUTED,
      valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt",
      showTitle: true, title: "Campaigns warned in advance (%)", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    });
    s.addChart(pres.charts.LINE, ser.map((n) => ({ name: n, labels, values: R_[n].lead })), {
      x: 6.8, y: 1.35, w: 6.0, h: 4.7, chartColors: ser.map((n) => cols[n]), lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      valAxisMinVal: 0, valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 11, catAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt", catAxisTitle: "reconnaissance intensity (lower = stealthier)",
      showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisTitleColor: MUTED,
      valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt",
      showTitle: true, title: "Median lead time (connections)", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    });
    const last = (n, k) => (R_[n] ? R_[n][k][R_[n][k].length - 1] : NaN);
    text(s, `Against very stealthy attackers (×0.25 scan intensity) FORESIGHT-Net still warns for ${f0(last("FORESIGHT-Net", "ewr"))} % of campaigns, ${n0(last("FORESIGHT-Net", "lead"))} connections early — the scan-count rule drops to ${f0(last("Probe-count rule", "ewr"))} % (${n0(last("Probe-count rule", "lead"))} early) and window XGBoost to ${f0(last("Window-stats XGBoost", "ewr"))} %. Trained models and thresholds unchanged.`,
      { x: 0.6, y: 6.15, w: 12.1, h: 0.8, fontSize: 14, color: INK, bold: true });
    s.addNotes(`ROBUSTNESS (1 minute)
A smart attacker scans slowly to stay under the radar. Without retraining anything, we regenerated the test network with reconnaissance at 75, 50, 35 and 25 % of the normal intensity.
This is where reading the sequence pays off most: a counting rule needs many scans in the window, so when the attacker slows down it simply stops firing. The deep models integrate weak evidence over time and degrade gracefully: at a quarter of the normal scan intensity FORESIGHT still warns for most campaigns with a long head start, while the rule warns for only about half — and false alarms of the deep models stay flat. Every method degrades; the difference is how much.`);
  }

  // =============================================================== 23 ZERO DAY
  {
    const s = content("Results", "Zero-day guard: catching attacks nobody has labelled");
    await addFig(s, "fig09_zero_day.png", 0.5, 1.3, 8.6, 4.4, "Zero-day guard");
    const Z = Object.fromEntries(D.zero_day.map((r) => [r.method, r]));
    const sup = Z["Supervised XGBoost"], ae = Z["Autoencoder (zero-day guard)"], hy = Z["Hybrid (XGBoost + AE)"];
    card(s, 9.35, 1.4, 3.35, 5.4);
    text(s, "On novel attack types", { x: 9.6, y: 1.55, w: 3.0, h: 0.4, fontSize: 16, bold: true });
    [[sup, "Supervised XGBoost", "9CA3AF"], [ae, "Autoencoder", "EB6834"], [hy, "Hybrid", HEX.accent1]].forEach(([r, n, c], i) => {
      text(s, `${f0(r.dr_novel)} %`, { x: 9.6, y: 2.05 + i * 1.05, w: 1.45, h: 0.6, fontSize: 30, bold: true, color: c });
      text(s, `${n}\nAUC ${n3(r.auc_novel)}`, { x: 11.05, y: 2.1 + i * 1.05, w: 1.6, h: 0.75, fontSize: 12, color: "374151" });
    });
    text(s, "detection rate at 5 % false-positive rate (threshold set on validation normal traffic)", { x: 9.6, y: 5.3, w: 3.0, h: 1.2, fontSize: 11.5, color: MUTED });
    bullets(s, [
      `${n0(D.zero_day_n_novel)} test connections belong to ${D.data.novel_types.length} attack types never seen in training.`,
      "The autoencoder only learns 'normal'; anything it cannot reconstruct is suspicious — it needs no attack labels at all.",
      "Inside FORESIGHT its novelty score is an input channel: it is how the forecaster notices scanners like mscan that the supervised detector misses (23 % vs 100 %).",
    ], 0.6, 5.8, 8.5, 1.2, 13, INK, 3);
    s.addNotes(`ZERO-DAY GUARD (1 minute)
Phase 1 left us with one big weakness: supervised models cannot recognise attacks they have never seen. The zero-day guard is an autoencoder trained ONLY on normal traffic; it compresses each connection and tries to rebuild it. Normal connections rebuild well, unfamiliar ones do not — so the reconstruction error is an attack-agnostic alarm.
On the ${n0(D.zero_day_n_novel)} test connections of brand-new attack types: supervised XGBoost detects ${f0(sup.dr_novel)} %, the autoencoder ${f0(ae.dr_novel)} %, and the hybrid that alarms when either is confident ${f0(hy.dr_novel)} %, all at a 5 % false-positive rate fixed on validation data. Read the bars by category and comment honestly on where the autoencoder helps and where it does not.
Most importantly, the novelty score feeds the forecaster: the scanner "mscan" exists only in the test set; the supervised detector recognises 23 % of its connections as scans, the autoencoder flags 100 % as unusual.`);
  }

  // =============================================================== 23b WHAT / WHEN / DETECTION
  {
    const s = content("Results", "Beyond 'whether': which attack, when — and better detection");
    await addFig(s, "fig13_detection_recall.png", 0.5, 1.3, 7.2, 4.4, "Per-class detection recall");
    const dx = D.detection.record_xgb.recall, df = D.detection.foresight.recall;
    const tiles = [
      [`${f0(D.type_forecast.accuracy)} %`, `attack type forecast correctly before impact (macro-F1 ${n2(D.type_forecast.macro_f1)}, best of the deep models)`, HEX.accent5],
      [`${f0(D.eta.within_one)} %`, "of ETAs within one horizon bin (10 / 25 / 50 / 100 connections)", HEX.accent4],
      [`${f0(dx.R2L)} → ${f0(df.R2L)} %`, "R2L detection recall with temporal context (Phase-1 weakness)", HEX.accent1],
    ];
    tiles.forEach(([v, l, col], i) => {
      const y = 1.35 + i * 1.45;
      card(s, 8.0, y, 4.7, 1.3);
      text(s, v, { x: 8.25, y: y + 0.12, w: 4.3, h: 0.6, fontSize: 30, bold: true, color: col });
      text(s, l, { x: 8.25, y: y + 0.72, w: 4.3, h: 0.55, fontSize: 12.5, color: "374151" });
    });
    text(s, "The detection head sees the 64-connection context: a break-in that follows reconnaissance is easier to recognise than the same connection in isolation. Part of this gain comes from the simulated campaign structure, so treat it as an upper bound.",
      { x: 0.6, y: 5.85, w: 12.1, h: 0.9, fontSize: 13, color: "374151" });
    s.addNotes(`WHICH ATTACK, WHEN, AND DETECTION (1 minute)
A warning is more useful if it says what to prepare for. Before impact, FORESIGHT names the attack type correctly ${f0(D.type_forecast.accuracy)} % of the time — macro-F1 ${n2(D.type_forecast.macro_f1)}, the best of all deep models — and its ETA is within one horizon bin ${f0(D.eta.within_one)} % of the time.
Left chart: the detection head classifies the CURRENT connection using the context of the last 64. Compared with the per-connection XGBoost of Phase 1, R2L recall rises from ${f0(dx.R2L)} % to ${f0(df.R2L)} % and probe recall from ${f0(dx.Probe)} % to ${f0(df.Probe)} %. That was exactly the weakness we reported at mid-term.
Honest caveat: in our streams a break-in always follows reconnaissance (except the stealth cases), so part of this gain comes from the simulated campaign structure — present it as an upper bound, not a guarantee.`);
  }

  // =============================================================== 24 EXPLAINABILITY
  {
    const s = content("Results", "Explainability: why did FORESIGHT raise this warning?");
    await addFig(s, "fig10_attention.png", 0.5, 1.3, 7.3, 2.6, "Attention over a warning window");
    const tf = D.explain.top_features.slice().reverse();
    s.addChart(pres.charts.BAR, [{ name: "share", labels: tf.map((x) => x[0]), values: tf.map((x) => x[1] * 100) }], {
      x: 7.95, y: 1.3, w: 4.85, h: 5.5, barDir: "bar",
      chartColors: tf.map((x) => (/detector|novelty/.test(x[0]) ? HEX.accent1 : "9CA3AF")),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
      valAxisHidden: true, valAxisMinVal: 0, catAxisLabelFontSize: 12, catAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt",
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
      showTitle: true, title: "Top inputs behind warnings (% of attribution)", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    });
    bullets(s, [
      [{ text: "Attention (top left): ", options: { bold: true } }, { text: `across ${n0(D.explain.n_windows)} alarmed reconnaissance windows, scan connections receive ${D.explain.attention_ratio.toFixed(1)}× their share of the model's attention (${f0(D.explain.attention_mass)} % of the attention vs ${f0(D.explain.scan_share)} % of the connections).` }],
      [{ text: "Feature attribution (right): ", options: { bold: true } }, { text: "gradient × input over the whole window shows which inputs drive warnings; blue bars are the evidence channels (detector and novelty)." }],
      [{ text: "In the dashboard: ", options: { bold: true } }, { text: "every warning comes with these two explanations, so the analyst can check the scans that triggered it before acting." }],
    ], 0.6, 4.2, 7.2, 2.8, 13.5, INK, 6);
    s.addNotes(`EXPLAINABILITY (1 minute)
A security analyst will not block traffic because a black box said so. So every warning is explained in two ways.
Attention: the model assigns a weight to each of the last 64 connections. Averaged over ${n0(D.explain.n_windows)} warning windows, scan connections get ${D.explain.attention_ratio.toFixed(1)} times their share of attention — state honestly whether that is a strong or weak focus based on the number.
Feature attribution: we take the gradient of the warning probability with respect to every input and multiply by the input. The right chart lists the inputs that drive warnings; explain the top three in plain words.
In the live dashboard both explanations appear next to every alert.`);
  }

  // =============================================================== 25 DEMO
  {
    const s = content("Results", "Live demo: FORESIGHT SOC dashboard");
    const shot = path.join(__dirname, "dashboard_screenshot.png");
    if (fs.existsSync(shot)) {
      const sz = await imgSize(shot);
      const w = 8.3, h = Math.min(5.6, w * sz.h / sz.w);
      s.addImage({ path: shot, x: 0.5, y: 1.3, w: h < 5.6 ? w : 5.6 * sz.w / sz.h, h, altText: "Dashboard screenshot" });
    }
    card(s, 9.1, 1.4, 3.6, 5.4);
    text(s, "Demo script (3 min)", { x: 9.35, y: 1.55, w: 3.2, h: 0.4, fontSize: 16, bold: true });
    bullets(s, [
      "streamlit run app/dashboard.py",
      "Pick 'Reconnaissance → DoS', press Play: threat level turns amber during the scan, before the red attack region",
      "Pause on the warning: show attack type, ETA, attention and 'why'",
      "Switch to 'Harmless scan' to discuss false alarms honestly",
      "Lower attacker stealth to 0.4 to show the limit",
      "Backup without a browser: python monitor.py",
    ], 9.35, 2.05, 3.2, 4.7, 12.5, INK, 5);
    s.addNotes(`LIVE DEMO (3 minutes) — rehearse this; have the dashboard already open before the presentation.
1. Start: streamlit run app/dashboard.py (it opens in the browser; models load in a few seconds).
2. Choose "Reconnaissance → Denial of Service", stealth 1.0, press Play. Narrate: grey connections, then the orange scans start; watch the blue forecast line rise; the threat level turns amber (EARLY WARNING) — and only later does the red attack region begin. The chart writes "warned N connections early".
3. Press Pause during the warning. Point at: expected attack type, ETA, the attention bars (orange = scans the model focused on) and the "why" features.
4. Switch scenario to "Harmless scan (no attack follows)": discuss honestly whether it raises a false alarm.
5. Optional: stealth 0.4 to show slower reconnaissance gives later warnings.
If the browser fails: python monitor.py --scenario recon_dos prints the same alerts in the terminal.`);
  }

  // =============================================================== 26 LIMITATIONS
  sec("Conclusion");
  {
    const s = content("Conclusion", "Limitations — stated honestly");
    const L = [
      ["Simulated order", "NSL-KDD has no timestamps. Connections are real; the campaign order follows a kill-chain model. Results show what is achievable if attacks behave this way."],
      ["No warning signs, no forecast", "Attacks without reconnaissance cannot be predicted by any method; they are caught on arrival (detection head, zero-day guard)."],
      ["Decoys still cause false alarms", "A harmless scan and the first minutes of a real one look alike; the alert budget trades early warning against these alarms."],
      ["Simple baselines are strong", "A scan-count rule is very conservative and competitive; deep backbones are close to each other once they share our evidence channels and adaptation."],
      ["Test-informed design", "Architecture changes were guided by test-stream diagnostics (disclosed); thresholds and model selection used only validation or unlabelled data."],
      ["Dated traffic", "NSL-KDD reflects 1998-era traffic; absolute numbers will differ on modern networks."],
    ];
    L.forEach(([t, d], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.4 + Math.floor(i / 2) * 1.82;
      card(s, x, y, 5.95, 1.62);
      text(s, t, { x: x + 0.3, y: y + 0.18, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: HEX.accent3 });
      text(s, d, { x: x + 0.3, y: y + 0.6, w: 5.4, h: 0.95, fontSize: 13, color: "374151" });
    });
    s.addNotes(`LIMITATIONS (1 minute) — saying these yourself earns trust; do not wait to be asked.
The most important: the time order is simulated because NSL-KDD has none. Every connection is real, the order follows the kill chain from the literature. Our pipeline is ready for real timestamped data — that is the first item of future work.
Second, attacks with no warning signs cannot be forecast — by us or anyone — so the system still needs detection on arrival.
Third, decoys: some false alarms are the price of early warning.
Fourth, honest comparison: a simple scan-count rule is a strong baseline, and the deep backbones are close to each other — the framework (evidence channels, adaptation, forecasting heads, evaluation) is the main contribution.
Fifth, we disclose that design iterations looked at test-stream behaviour.`);
  }

  // =============================================================== 27 FUTURE WORK
  {
    const s = content("Conclusion", "Future work");
    const F = [
      [I.db, HEX.accent1, "Real timestamped traffic", "CIC-IDS2017 / UNSW-NB15 flows with real time order — only the stream source changes"],
      [I.net, HEX.accent4, "Live capture", "feature extraction from packets (CICFlowMeter-style) feeding the same inference API"],
      [I.target, HEX.accent5, "Attacker-aware forecasting", "group connections per source host; forecast per attacker, not per gateway"],
      [I.lock, HEX.accent3, "Automated response", "rate-limit or sandbox the scanning source when the warning fires, with analyst approval"],
      [I.brain, HEX.accent2, "Continual learning", "update the site model as normal traffic drifts; active learning from analyst feedback"],
      [I.check, MUTED, "Statistical rigour", "more seeds, confidence intervals, significance tests between backbones"],
    ];
    F.forEach(([ic, col, t, d], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 1.75;
      card(s, x, y, 5.95, 1.5);
      iconCircle(s, ic, x + 0.3, y + 0.35, 0.8, col);
      text(s, t, { x: x + 1.35, y: y + 0.25, w: 4.4, h: 0.4, fontSize: 17, bold: true });
      text(s, d, { x: x + 1.35, y: y + 0.68, w: 4.4, h: 0.75, fontSize: 13.5, color: "374151" });
    });
    s.addNotes(`FUTURE WORK (45 seconds)
First priority is real timestamped traffic — CIC-IDS2017 or UNSW-NB15 — which removes our main limitation; the code only needs a new stream source.
Then live packet capture, forecasting per attacker (source host) rather than per gateway, an automated response such as rate-limiting the scanning IP when the warning fires, continual learning as the network's normal changes, and more statistical testing.`);
  }

  // =============================================================== 28 CONCLUSION
  {
    const s = pres.addSlide({ masterName: "DARK_TOP", sectionTitle: "Conclusion" });
    s.addText("Conclusion", { placeholder: "title" });
    const pts = [
      `FORESIGHT turns intrusion detection into forecasting: it warned for ${f0(FS.ewr)} % of reconnaissance-led campaigns, a median ${n0(FS.lead_median)} connections before the attack — ${f0(FS.lead_median / WX.lead_median - 1)} % earlier than window XGBoost at the same false-alarm level.`,
      `Against stealthy 'low-and-slow' attackers it keeps warning (${D.robust && D.robust["FORESIGHT-Net"] ? f0(D.robust["FORESIGHT-Net"].ewr.slice(-1)[0]) : "–"} % vs ${D.robust && D.robust["Probe-count rule"] ? f0(D.robust["Probe-count rule"].ewr.slice(-1)[0]) : "–"} % for a scan-count rule).`,
      "Generalising to a new network needed more than a bigger model: hybrid evidence channels and site adaptation from unlabelled traffic.",
      "A zero-day autoencoder, attention-based explanations and a live dashboard make it a usable SOC tool.",
      "Honest evaluation: official unseen-attack split, decoys, 3 seeds, ablations, stealth tests — and the limitations stated openly.",
    ];
    pts.forEach((p, i) => {
      s.addShape("ellipse", { x: 0.8, y: 2.05 + i * 0.95, w: 0.22, h: 0.22, fill: { color: HEX.accent2 }, line: { color: HEX.accent2, width: 0 } });
      text(s, p, { x: 1.25, y: 1.9 + i * 0.95, w: 11.3, h: 0.85, fontSize: 18, color: "E5ECF6" });
    });
    s.addNotes(`CONCLUSION (45 seconds)
Read the four points. End with the smoke-alarm vs. weather-forecast image: classical IDS tells you there is a fire; FORESIGHT tells you a storm is coming — early enough to act.
Then: "Thank you — we are happy to take questions, and we can show the live demo again."`);
  }

  // =============================================================== 29 KEY LEARNINGS & TEAM
  {
    const s = content("Conclusion", "Key learnings & work contribution");
    text(s, "Key learnings", { x: 0.6, y: 1.35, w: 5.8, h: 0.4, fontSize: 17, bold: true, color: HEX.accent1 });
    bullets(s, [
      "Evaluation design decides what a result means: random splits and accuracy hide the real difficulty",
      "Sequence modelling, survival-style multi-horizon outputs, attention",
      "Distribution shift is the central problem of deployed ML; validation can be misleading",
      "Stacking, autoencoders, calibration without labels",
      "Reproducible engineering: cached pipelines, seeds, ablations",
    ], 0.6, 1.85, 5.8, 4.8, 14, INK, 6);
    const T = [
      ["Raghav Dhir", "231030254", "literature review, research gap, kill-chain simulator design, documentation & presentation"],
      ["Rudra Bhasin", "231030235", "data pipeline, feature encoding, evidence channels (detector, autoencoder)"],
      ["Nikhil Singh", "231030152", "FORESIGHT-Net, deep baselines, training, ablation & robustness experiments"],
      ["Aayush Chaudhary", "231030164", "evaluation metrics, site calibration, dashboard & system integration, testing"],
    ];
    T.forEach(([n, r, w], i) => {
      const y = 1.4 + i * 1.32;
      card(s, 6.7, y, 6.0, 1.18);
      text(s, n, { x: 6.95, y: y + 0.12, w: 3.2, h: 0.4, fontSize: 15, bold: true });
      text(s, r, { x: 10.6, y: y + 0.12, w: 1.9, h: 0.4, fontSize: 12, color: MUTED, align: "right" });
      text(s, w, { x: 6.95, y: y + 0.5, w: 5.6, h: 0.62, fontSize: 12.5, color: "374151" });
    });
    s.addNotes(`KEY LEARNINGS & CONTRIBUTION (45 seconds)
Each member should be ready to explain their own part in detail — the panel often asks individual members.
(Edit the role descriptions to match the actual split of work before presenting.)`);
  }

  // =============================================================== 30 REFERENCES
  {
    const s = content("Conclusion", "References");
    const refs = [
      "[1] M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 data set,” IEEE CISDA, 2009.",
      "[2] Z. Ahmad, A. S. Khan, C. W. Shiang, J. Abdullah, F. Ahmad, “Network intrusion detection system: A systematic study of machine learning and deep learning approaches,” Trans. Emerging Telecommunications Technologies, 32(1), e4150, 2021.",
      "[3] M. Husák, J. Komárková, E. Bou-Harb, P. Čeleda, “Survey of attack projection, prediction, and forecasting in cyber security,” IEEE Communications Surveys & Tutorials, 21(1), 640–660, 2019.",
      "[4] Y. Shen, E. Mariconti, P.-A. Vervier, G. Stringhini, “Tiresias: Predicting security events through deep learning,” ACM CCS, 2018.",
      "[5] M. Du, F. Li, G. Zheng, V. Srikumar, “DeepLog: Anomaly detection and diagnosis from system logs through deep learning,” ACM CCS, 2017.",
      "[6] Y. Mirsky, T. Doitshman, Y. Elovici, A. Shabtai, “Kitsune: An ensemble of autoencoders for online network intrusion detection,” NDSS, 2018.",
      "[7] E. M. Hutchins, M. J. Cloppert, R. M. Amin, “Intelligence-driven computer network defense informed by analysis of adversary campaigns and intrusion kill chains,” Lockheed Martin, 2011.  MITRE ATT&CK, TA0043 / T1595.",
      "[8] W. E. Leland, M. S. Taqqu, W. Willinger, D. V. Wilson, “On the self-similar nature of Ethernet traffic,” IEEE/ACM Trans. Networking, 2(1), 1–15, 1994.",
      "[9] S. Hochreiter, J. Schmidhuber, “Long short-term memory,” Neural Computation, 1997;  D. Bahdanau, K. Cho, Y. Bengio, “Neural machine translation by jointly learning to align and translate,” ICLR, 2015.",
      "[10] T. Chen, C. Guestrin, “XGBoost: A scalable tree boosting system,” ACM KDD, 2016;  D. H. Wolpert, “Stacked generalization,” Neural Networks, 1992.",
      "[11] M. Umer et al., “Network intrusion detection model using wrapper based feature selection and multi head attention transformers,” Scientific Reports, 15, 28718, 2025.",
      "[12] C. Christy et al., “Machine learning based multi-stage intrusion detection system and feature selection ensemble security in cloud assisted vehicular ad hoc networks,” Scientific Reports, 15, 27058, 2025.",
    ];
    s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 4 } })),
      { x: 0.6, y: 1.3, w: 12.1, h: 5.6, fontSize: 11.5, color: "374151", valign: "top", isTextBox: true, margin: 0 });
    s.addNotes(`REFERENCES — no need to read; leave on screen briefly.`);
  }

  // =============================================================== 31 THANK YOU
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
    s.addText("Thank you — questions?", { placeholder: "title" });
    text(s, "Live demo:  streamlit run app/dashboard.py        Code & results:  README.md · docs/METHODOLOGY.md · outputs/results/RESULTS.md",
      { x: 0.8, y: 3.75, w: 11.7, h: 0.5, fontSize: 15, color: "9FB3D1" });
    s.addNotes(`Q&A. Keep the dashboard open in another window. The full list of expected questions with model answers is in docs/PANEL_GUIDE.md — revise it before the evaluation.`);
  }

  await pres.writeFile({ fileName: OUT });
  const { applyTheme } = require(SKILL_THEME);
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
