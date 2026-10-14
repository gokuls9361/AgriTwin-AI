/**
 * AgriTwin AI — Modular Interactive Digital Twin Single-Page Application (SPA)
 * Architecture:
 *   - Concept Navigation Controller (One concept in one dedicated place)
 *   - Live Real-Time Auto-Stream Simulation Clock
 *   - Procedural 2D/3D Biome & Multi-Layer Soil Strata Canvas
 *   - AI Multi-Model Inference Suite (RF, SVM, pure NumPy LSTM)
 *   - 4-Zone Spatial Explorer & Comparative Soil Hydrology
 *   - Prescriptive Smart Irrigation & Carbon/Tariff Optimizer
 *   - Weather Stress Sandbox & What-If Simulator
 *   - Web Audio Procedural Acoustic Synthesizer
 */

// Global Digital Twin State
const state = {
  activeConcept: "twin",
  activeZone: 2,
  autoStream: false,
  streamTick: 0,
  seasonDay: 42,
  hour: 14,
  minute: 25,
  moisture: 26.5,
  temperature: 28.0,
  humidity: 58.0,
  vpd: 1.60,
  rain24h: 1.5,
  gdd: 620,
  stage: "3_Flowering",
  ndvi: 0.78,
  ph: 6.6,
  injectStress: false,
  crashSeverity: 12.0,
  soundEnabled: false,
  animTick: 0
};

// Spatial Micro-Plot Profiles
const zones = {
  1: {
    name: "Zone 1: Sandy Loam (West Plot)",
    badge: "Zone 1: Sandy Loam",
    baseMoist: 17.5,
    baseTemp: 29.5,
    baseVpd: 1.95,
    baseNdvi: 0.68,
    basePh: 6.2,
    soilDesc: "Coarse Sand (22% Field Capacity)",
    priority: "HIGH (Fast Drainage)",
    priorityColor: "#d97706",
    soilTone: [100, 75, 40]
  },
  2: {
    name: "Zone 2: Clay Loam (East Plot - Primary)",
    badge: "Zone 2: Clay Loam",
    baseMoist: 26.5,
    baseTemp: 28.0,
    baseVpd: 1.60,
    baseNdvi: 0.78,
    basePh: 6.6,
    soilDesc: "Clay Loam (34% Field Capacity)",
    priority: "NOMINAL",
    priorityColor: "#10b981",
    soilTone: [65, 42, 22]
  },
  3: {
    name: "Zone 3: Lowland Alluvial (South Basin)",
    badge: "Zone 3: Alluvial Basin",
    baseMoist: 34.0,
    baseTemp: 25.5,
    baseVpd: 1.15,
    baseNdvi: 0.82,
    basePh: 7.1,
    soilDesc: "Alluvial Silt (38% Field Capacity)",
    priority: "LOW (Waterlogging Risk)",
    priorityColor: "#0284c7",
    soilTone: [35, 24, 15]
  },
  4: {
    name: "Zone 4: Sun-Exposed Ridge (North Slope)",
    badge: "Zone 4: Ridge",
    baseMoist: 19.5,
    baseTemp: 32.5,
    baseVpd: 2.40,
    baseNdvi: 0.64,
    basePh: 6.4,
    soilDesc: "Rocky Ridge (28% Field Capacity)",
    priority: "URGENT (Solar Heat)",
    priorityColor: "#ef4444",
    soilTone: [90, 50, 25]
  }
};

// --- Web Audio Synthesizer ---
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
}

function playTone(freq, type = "sine", duration = 0.12, gain = 0.08) {
  if (!state.soundEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gNode = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gNode.gain.setValueAtTime(gain, audioCtx.currentTime);
    gNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gNode);
    gNode.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

function playWaterDrop() {
  playTone(880, "sine", 0.08, 0.05);
  setTimeout(() => playTone(1200, "sine", 0.06, 0.03), 40);
}

// --- Concept Tab Navigation ---
function switchConcept(conceptId) {
  state.activeConcept = conceptId;
  playTone(540, "sine", 0.08, 0.04);

  // Update tab buttons
  document.querySelectorAll(".concept-tab-btn").forEach(btn => {
    if (btn.getAttribute("data-concept") === conceptId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update view panels
  document.querySelectorAll(".concept-panel").forEach(panel => {
    if (panel.id === `view-${conceptId}`) {
      panel.classList.add("active");
    } else {
      panel.classList.remove("active");
    }
  });

  // Smooth scroll to top of app container
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// --- Background Neural Particle Animation ---
function setupBgParticles() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });

  const particles = Array.from({ length: 40 }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    r: Math.random() * 2 + 1,
    alpha: Math.random() * 0.35 + 0.1
  }));

  function drawBg() {
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.fillStyle = `rgba(16, 185, 129, ${p.alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 120) {
          ctx.strokeStyle = `rgba(16, 185, 129, ${(1 - dist / 120) * 0.1})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(drawBg);
  }
  drawBg();
}

// --- Procedural Biome Canvas ---
function setupBiomeCanvas() {
  const canvas = document.getElementById("biome-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const rainDrops = Array.from({ length: 35 }, () => ({
    x: Math.random() * rect.width,
    y: Math.random() * (rect.height * 0.45),
    vy: Math.random() * 4 + 3,
    len: Math.random() * 8 + 6
  }));

  function renderBiome() {
    state.animTick++;
    const w = rect.width;
    const h = rect.height;
    const groundY = h * 0.46;

    ctx.clearRect(0, 0, w, h);

    // 1. Sky & Solar Radiation
    const tempIntensity = Math.min(1.0, Math.max(0.0, (state.temperature - 15) / 30));
    const skyGrad = ctx.createLinearGradient(0, 0, 0, groundY);
    skyGrad.addColorStop(0, `rgb(${Math.round(11 + tempIntensity * 45)}, ${Math.round(34 + (1 - tempIntensity) * 20)}, 22)`);
    skyGrad.addColorStop(1, `rgb(${Math.round(23 + tempIntensity * 30)}, 56, 38)`);
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, w, groundY);

    const sunX = w * 0.84;
    const sunY = 38;
    const sunGrad = ctx.createRadialGradient(sunX, sunY, 4, sunX, sunY, 36);
    sunGrad.addColorStop(0, `rgba(255, 225, 110, ${0.75 + tempIntensity * 0.25})`);
    sunGrad.addColorStop(1, "rgba(255, 225, 110, 0)");
    ctx.fillStyle = sunGrad;
    ctx.beginPath();
    ctx.arc(sunX, sunY, 36, 0, Math.PI * 2);
    ctx.fill();

    // 2. Rain Animation
    if (state.rain24h > 1.5) {
      ctx.strokeStyle = "rgba(147, 197, 253, 0.65)";
      ctx.lineWidth = 1.3;
      rainDrops.forEach(d => {
        d.y += d.vy;
        if (d.y > groundY) d.y = 0;
        ctx.beginPath();
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(d.x - 1, d.y + d.len);
        ctx.stroke();
      });
    }

    // 3. Multi-Layer Soil Strata
    const zInfo = zones[state.activeZone] || zones[2];
    const effMoist = state.moisture - (state.injectStress ? state.crashSeverity : 0);
    const moistRatio = Math.min(1.0, Math.max(0.1, effMoist / 45.0));
    const [tr, tg, tb] = zInfo.soilTone;

    // Topsoil (0-10cm)
    ctx.fillStyle = `rgb(${Math.round(tr + (1 - moistRatio) * 30)}, ${Math.round(tg + (1 - moistRatio) * 20)}, ${Math.round(tb + (1 - moistRatio) * 15)})`;
    ctx.fillRect(0, groundY, w, (h - groundY) * 0.28);

    // Root Zone (10-30cm)
    ctx.fillStyle = `rgb(${Math.round(tr - moistRatio * 20)}, ${Math.round(tg - moistRatio * 15)}, ${Math.round(tb - moistRatio * 10)})`;
    ctx.fillRect(0, groundY + (h - groundY) * 0.28, w, (h - groundY) * 0.42);

    // Deep Subsoil (30-60cm)
    ctx.fillStyle = `rgb(${Math.round(tr * 0.65 - moistRatio * 10)}, ${Math.round(tg * 0.65 - moistRatio * 10)}, ${Math.round(tb * 0.65 - moistRatio * 5)})`;
    ctx.fillRect(0, groundY + (h - groundY) * 0.70, w, (h - groundY) * 0.30);

    // Layer Dividers
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, groundY + (h - groundY) * 0.28);
    ctx.lineTo(w, groundY + (h - groundY) * 0.28);
    ctx.moveTo(0, groundY + (h - groundY) * 0.70);
    ctx.lineTo(w, groundY + (h - groundY) * 0.70);
    ctx.stroke();

    // Strata Labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
    ctx.font = "9px sans-serif";
    ctx.fillText("Topsoil (0-10cm)", 10, groundY + 16);
    ctx.fillText("Active Root Zone (10-30cm)", 10, groundY + (h - groundY) * 0.28 + 16);
    ctx.fillText("Deep Subsoil (30-60cm)", 10, groundY + (h - groundY) * 0.70 + 16);

    // 4. Procedural Crop Plant (Scales dynamically with Season Day)
    const plantX = w * 0.50;
    const growthScale = Math.min(1.0, Math.max(0.3, state.seasonDay / 50.0));
    const sway = Math.sin(state.animTick * 0.03) * 4.5 * growthScale;
    const plantHealth = Math.max(0.2, Math.min(1.0, state.ndvi));
    const stemHeight = 85 * growthScale;

    // Stem
    ctx.strokeStyle = `rgb(${Math.round(40 + (1 - plantHealth) * 120)}, ${Math.round(160 * plantHealth)}, 50)`;
    ctx.lineWidth = Math.max(2, 4 * growthScale);
    ctx.beginPath();
    ctx.moveTo(plantX, groundY);
    ctx.quadraticCurveTo(plantX + sway * 0.5, groundY - stemHeight * 0.5, plantX + sway, groundY - stemHeight);
    ctx.stroke();

    // Leaves
    ctx.fillStyle = `rgb(${Math.round(40 + (1 - plantHealth) * 140)}, ${Math.round(180 * plantHealth)}, 60)`;
    [-1, 1].forEach(side => {
      ctx.beginPath();
      ctx.ellipse(plantX + side * (22 * growthScale) + sway * 0.7, groundY - stemHeight * 0.55, 20 * growthScale, 7 * growthScale, side * 0.4, 0, Math.PI * 2);
      ctx.fill();

      if (growthScale > 0.5) {
        ctx.beginPath();
        ctx.ellipse(plantX + side * (18 * growthScale) + sway * 0.9, groundY - stemHeight * 0.85, 16 * growthScale, 6 * growthScale, side * 0.35, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Golden Flower / Blossom (Only when flowering or later)
    if (state.stage.includes("Flowering") || state.stage.includes("GrainFilling") || state.stage.includes("Maturity")) {
      ctx.fillStyle = state.stage.includes("Maturity") ? "#d97706" : "#fbbf24";
      ctx.beginPath();
      ctx.arc(plantX + sway, groundY - stemHeight - 7, 9 * growthScale, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(plantX + sway, groundY - stemHeight - 7, 5 * growthScale, 0, Math.PI * 2);
      ctx.fill();
    }

    // Roots
    ctx.strokeStyle = "rgba(229, 231, 235, 0.8)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(plantX, groundY);
    ctx.lineTo(plantX - 14, groundY + 28);
    ctx.moveTo(plantX, groundY);
    ctx.lineTo(plantX + 18, groundY + 34);
    ctx.moveTo(plantX, groundY);
    ctx.lineTo(plantX - 4, groundY + 54);
    ctx.stroke();

    requestAnimationFrame(renderBiome);
  }
  renderBiome();
}

// --- Live Telemetry Inference & Calculations ---
function runInference() {
  const z = zones[state.activeZone];
  let effMoisture = state.moisture;
  let effTemp = state.temperature;
  let effVpd = state.vpd;
  let effNdvi = state.ndvi;

  if (state.injectStress) {
    effMoisture = Math.max(5.0, effMoisture - state.crashSeverity);
    effNdvi = Math.max(0.15, effNdvi - 0.12);
  }

  // 1. Random Forest Classifier
  const moistureZ = (20.0 - effMoisture) / 4.5;
  const rainZ = (3.0 - state.rain24h) / 4.0;
  const vpdZ = (effVpd - 1.5) / 0.8;
  const stageWeight = state.stage.includes("Flowering") ? 0.45 : 0.0;
  const score = 1.2 * moistureZ + 0.4 * rainZ + 0.2 * vpdZ + stageWeight;
  const rfProb = 1.0 / (1.0 + Math.exp(-score));
  const rfPred = rfProb >= 0.5 ? 1 : 0;
  const rfConf = Math.max(rfProb, 1.0 - rfProb);

  // 2. SVM Anomaly Detector
  const anomScore = (effTemp > 31.0 ? 1.6 : 0.0) + (effMoisture < 19.0 ? 2.2 : 0.0) + (state.injectStress ? 2.5 : 0.0);
  const svmProb = 1.0 / (1.0 + Math.exp(-(anomScore - 1.2)));
  const svmPred = svmProb >= 0.5 ? 1 : 0;
  const svmConf = Math.max(svmProb, 1.0 - svmProb);

  // 3. LSTM 3-Day Forecast
  const lstmForecast = Math.max(7.5, effMoisture - 1.9 * (effVpd / 1.5) + (state.rain24h > 8 ? 4.5 : 0));
  const lstmUnc = 0.85 + (effTemp > 31.0 ? 0.6 : 0.1) + (state.injectStress ? 1.2 : 0.0);

  // 4. Prescriptive Calculations
  const targetMoist = 28.0;
  const deficit = Math.max(0, targetMoist - effMoisture);
  const waterReqMm = deficit * 2.8;
  const waterM3Ha = waterReqMm * 10.0;
  const pumpHours = waterM3Ha / 45.0;
  const energyKwh = waterM3Ha * 1.2;
  const co2Kg = energyKwh * 0.45;
  const costOffpeak = energyKwh * 0.08;

  // DOM Updates (All synced views)
  document.querySelectorAll(".val-moisture").forEach(el => el.innerText = effMoisture.toFixed(1) + "%");
  document.querySelectorAll(".val-temp").forEach(el => el.innerText = effTemp.toFixed(1) + "°C");
  document.querySelectorAll(".val-vpd").forEach(el => el.innerText = effVpd.toFixed(2) + " kPa");
  document.querySelectorAll(".val-ndvi").forEach(el => el.innerText = effNdvi.toFixed(2));
  document.querySelectorAll(".val-gdd").forEach(el => el.innerText = state.gdd + " °C-d");
  document.querySelectorAll(".val-ph").forEach(el => el.innerText = state.ph.toFixed(1));

  // Gauges Progress Bars
  document.querySelectorAll(".bar-moist").forEach(el => el.style.width = Math.min(100, Math.max(5, (effMoisture / 50) * 100)) + "%");
  document.querySelectorAll(".bar-temp").forEach(el => el.style.width = Math.min(100, Math.max(5, (effTemp / 48) * 100)) + "%");
  document.querySelectorAll(".bar-vpd").forEach(el => el.style.width = Math.min(100, Math.max(5, (effVpd / 3.5) * 100)) + "%");
  document.querySelectorAll(".bar-ndvi").forEach(el => el.style.width = (effNdvi * 100).toFixed(0) + "%");
  document.querySelectorAll(".bar-ph").forEach(el => el.style.width = Math.min(100, Math.max(10, (state.ph / 10.0) * 100)) + "%");

  // RF Box Updates
  document.querySelectorAll(".rf-badge").forEach(el => {
    if (rfPred === 1) {
      el.className = "ai-tag tag-irrigate rf-badge";
      el.innerText = "🚨 IRRIGATION REQUIRED";
    } else {
      el.className = "ai-tag tag-optimal rf-badge";
      el.innerText = "✅ NO IRRIGATION NEEDED";
    }
  });
  document.querySelectorAll(".rf-bar").forEach(el => el.style.width = (rfConf * 100).toFixed(0) + "%");
  document.querySelectorAll(".rf-conf-text").forEach(el => el.innerText = `Tree-Vote Consensus: ${(rfConf * 100).toFixed(1)}%`);

  // SVM Box Updates
  document.querySelectorAll(".svm-badge").forEach(el => {
    if (svmPred === 1) {
      el.className = "ai-tag tag-stress svm-badge";
      el.innerText = "🔴 STRESS ANOMALY DETECTED";
    } else {
      el.className = "ai-tag tag-normal svm-badge";
      el.innerText = "🟢 CANOPY STATE NORMAL";
    }
  });
  document.querySelectorAll(".svm-bar").forEach(el => el.style.width = (svmConf * 100).toFixed(0) + "%");
  document.querySelectorAll(".svm-conf-text").forEach(el => el.innerText = `Platt-Scaled Confidence: ${(svmConf * 100).toFixed(1)}%`);

  // LSTM Box Updates
  document.querySelectorAll(".lstm-val").forEach(el => el.innerText = `${lstmForecast.toFixed(1)}% ± ${lstmUnc.toFixed(2)}%`);
  const lstmConfRatio = Math.max(0.2, 1.0 - lstmUnc / 6.0);
  document.querySelectorAll(".lstm-bar").forEach(el => el.style.width = (lstmConfRatio * 100).toFixed(0) + "%");
  document.querySelectorAll(".lstm-conf-text").forEach(el => el.innerText = `MC-Dropout Uncertainty Band: ±${lstmUnc.toFixed(2)}%`);

  // Prescriptive Optimizer Updates (Indian Agricultural Tariff: INR 3.25/kWh off-peak)
  const costOffpeakInr = energyKwh * 3.25;
  document.querySelectorAll(".adv-water").forEach(el => el.innerText = `${waterReqMm.toFixed(1)} mm (${waterM3Ha.toFixed(1)} m³/ha)`);
  document.querySelectorAll(".adv-pump").forEach(el => el.innerText = `${pumpHours.toFixed(2)} hrs (${Math.round(pumpHours * 60)} mins)`);
  document.querySelectorAll(".adv-energy").forEach(el => el.innerText = `${energyKwh.toFixed(1)} kWh`);
  document.querySelectorAll(".adv-co2").forEach(el => el.innerText = `${co2Kg.toFixed(1)} kg CO₂e`);
  document.querySelectorAll(".adv-cost").forEach(el => el.innerText = `₹${costOffpeakInr.toFixed(2)}`);

  // Update Spatial Tiles with consistent current root-zone moisture and layer info
  for (let zid = 1; zid <= 4; zid++) {
    const tileEl = document.getElementById(`tile-m-${zid}`);
    if (tileEl) {
      const zOffset = (zid === 1) ? -2.5 : (zid === 3) ? 3.5 : (zid === 4) ? -1.8 : 0.0;
      const zMoist = Math.max(5.0, Math.min(50.0, effMoisture + zOffset));
      tileEl.innerText = zMoist.toFixed(1) + "%";
    }
  }
}

// Select Zone Handler
function selectZone(zId) {
  const z = zones[zId];
  if (!z) return;

  state.activeZone = zId;
  state.moisture = z.baseMoist;
  state.temperature = z.baseTemp;
  state.vpd = z.baseVpd;
  state.ndvi = z.baseNdvi;
  state.ph = z.basePh;

  const sMoist = document.getElementById("slider-moist");
  const sTemp = document.getElementById("slider-temp");
  if (sMoist) sMoist.value = state.moisture;
  if (sTemp) sTemp.value = state.temperature;

  const lblMoist = document.getElementById("label-moist");
  const lblTemp = document.getElementById("label-temp");
  if (lblMoist) lblMoist.innerText = state.moisture.toFixed(1) + "%";
  if (lblTemp) lblTemp.innerText = state.temperature.toFixed(1) + "°C";

  // Badges & text
  document.querySelectorAll(".active-zone-label").forEach(el => el.innerText = z.name);
  const biomeBadge = document.getElementById("biome-zone-badge");
  if (biomeBadge) biomeBadge.innerText = z.badge;
  const biomeSoil = document.getElementById("biome-soil-type");
  if (biomeSoil) biomeSoil.innerHTML = `Soil: <strong>${z.soilDesc}</strong>`;
  const biomePri = document.getElementById("biome-priority");
  if (biomePri) biomePri.innerHTML = `Priority: <strong style="color:${z.priorityColor};">${z.priority}</strong>`;

  // Update active pill & tile classes
  document.querySelectorAll(".zone-pill").forEach(p => {
    if (parseInt(p.getAttribute("data-zone")) === zId) {
      p.classList.add("active");
    } else {
      p.classList.remove("active");
    }
  });

  document.querySelectorAll(".zone-interactive-card").forEach(t => {
    if (t.id === `zone-tile-${zId}`) {
      t.classList.add("active");
    } else {
      t.classList.remove("active");
    }
  });

  // Flash highlight animation
  document.querySelectorAll(".dial-card").forEach(c => {
    c.classList.add("dial-flash");
    setTimeout(() => c.classList.remove("dial-flash"), 600);
  });

  playTone(480 + zId * 60, "sine", 0.12, 0.08);
  runInference();
}

// --- Phenological & Biological Curves by Season Day ---
function getStageForDay(day) {
  if (day <= 10) return "1_Emergence";
  if (day <= 35) return "2_Vegetative";
  if (day <= 60) return "3_Flowering";
  if (day <= 80) return "4_GrainFilling";
  return "5_Maturity";
}

function getGddForDay(day) {
  return Math.round(day * 14.8);
}

function getNdviForDay(day) {
  if (day <= 10) return 0.25 + (day / 10) * 0.15;
  if (day <= 35) return 0.40 + ((day - 10) / 25) * 0.35;
  if (day <= 60) return 0.75 + ((day - 35) / 25) * 0.08;
  if (day <= 80) return 0.83 - ((day - 60) / 20) * 0.18;
  return Math.max(0.42, 0.65 - ((day - 80) / 10) * 0.22);
}

// --- Historical Past 7 Days Table Renderer ---
function renderHistoryTable() {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;

  const z = zones[state.activeZone] || zones[2];
  const currDay = state.seasonDay;
  const startDay = Math.max(1, currDay - 6);
  let rowsHtml = "";

  for (let d = startDay; d <= currDay; d++) {
    const isCurrent = d === currDay;
    const stage = getStageForDay(d);
    const gddVal = getGddForDay(d);
    const pastMoist = (z.baseMoist + Math.sin(d * 0.45) * 2.8).toFixed(1);
    const pastTemp = (z.baseTemp + Math.cos(d * 0.3) * 1.8).toFixed(1);
    const pastRain = (d % 8 === 0 ? 14.2 : d % 4 === 0 ? 4.5 : 0.0).toFixed(1);

    const isIrrigation = parseFloat(pastMoist) < 22.0 || stage === "3_Flowering" && parseFloat(pastMoist) < 25.0;
    const badge = isIrrigation 
      ? '<span style="color:#dc2626; font-weight:800;">🚨 Irrigated (12mm)</span>' 
      : '<span style="color:#059669; font-weight:700;">✅ Nominal (No Deficit)</span>';

    rowsHtml += `
      <tr class="${isCurrent ? 'current-day-row' : ''}">
        <td><strong>Day ${d}</strong> ${isCurrent ? '<span style="background:#10b981; color:#042f1d; padding:2px 6px; border-radius:10px; font-size:0.75rem; font-weight:800; margin-left:4px;">TODAY</span>' : ''}</td>
        <td><code>${stage}</code></td>
        <td><strong>${pastMoist}%</strong></td>
        <td>${pastTemp}°C</td>
        <td>${pastRain} mm</td>
        <td>${gddVal} °C-d</td>
        <td>${badge}</td>
      </tr>
    `;
  }

  tbody.innerHTML = rowsHtml;
}

// --- Global Clock & Timeline Controller ---
function updateClockDisplay() {
  const clockText = document.getElementById("clock-timestamp");
  const scrubber = document.getElementById("season-scrubber");
  const scrubberLabel = document.getElementById("scrubber-label");
  const clockStageBadge = document.getElementById("clock-stage-badge");
  const biomeStageText = document.getElementById("biome-stage-text");

  const pad = n => String(n).padStart(2, "0");
  if (clockText) {
    clockText.innerText = `Season Day ${state.seasonDay}/90 • ${pad(state.hour)}:${pad(state.minute)} UTC`;
  }
  if (scrubber) scrubber.value = state.seasonDay;
  if (scrubberLabel) scrubberLabel.innerText = `Day ${state.seasonDay}`;

  state.stage = getStageForDay(state.seasonDay);
  state.gdd = getGddForDay(state.seasonDay);
  state.ndvi = getNdviForDay(state.seasonDay);

  if (clockStageBadge) clockStageBadge.innerText = state.stage;
  if (biomeStageText) biomeStageText.innerText = state.stage;

  // Daily diurnal temperature / VPD fluctuations
  const diurnal = Math.sin((state.hour - 8) * (Math.PI / 12));
  state.temperature = zones[state.activeZone].baseTemp + diurnal * 4.5;
  state.vpd = Math.max(0.6, zones[state.activeZone].baseVpd + diurnal * 0.7);

  // Subterranean moisture dynamics across the 90-day season
  const dayOscillation = Math.sin(state.seasonDay * 0.45) * 3.2;
  const hourDrying = (state.hour >= 10 && state.hour <= 18) ? -1.1 : 0.3;
  state.moisture = Math.max(8.0, Math.min(42.0, zones[state.activeZone].baseMoist + dayOscillation + hourDrying));

  // Rain dynamics
  state.rain24h = (state.seasonDay % 8 === 0) ? 14.5 : (state.seasonDay % 4 === 0) ? 4.2 : 0.5;

  // Sync sandbox slider labels
  const sMoist = document.getElementById("slider-moist");
  const sTemp = document.getElementById("slider-temp");
  const sRain = document.getElementById("slider-rain");
  const lblMoist = document.getElementById("label-moist");
  const lblTemp = document.getElementById("label-temp");
  const lblRain = document.getElementById("label-rain");
  if (sMoist) sMoist.value = state.moisture;
  if (sTemp) sTemp.value = state.temperature;
  if (sRain) sRain.value = state.rain24h;
  if (lblMoist) lblMoist.innerText = state.moisture.toFixed(1) + "%";
  if (lblTemp) lblTemp.innerText = state.temperature.toFixed(1) + "°C";
  if (lblRain) lblRain.innerText = state.rain24h.toFixed(1) + " mm";

  // Flash highlight
  document.querySelectorAll(".dial-card").forEach(c => {
    c.classList.add("dial-flash");
    setTimeout(() => c.classList.remove("dial-flash"), 400);
  });

  renderHistoryTable();
  runInference();
}

function advanceTime(minutes = 5) {
  state.minute += minutes;
  
  // Handle forward minute overflow
  while (state.minute >= 60) {
    state.minute -= 60;
    state.hour += 1;
  }
  // Handle backward minute underflow
  while (state.minute < 0) {
    state.minute += 60;
    state.hour -= 1;
  }

  // Handle forward hour overflow
  while (state.hour >= 24) {
    state.hour -= 24;
    state.seasonDay += 1;
    if (state.seasonDay > 90) state.seasonDay = 1;
  }
  // Handle backward hour underflow
  while (state.hour < 0) {
    state.hour += 24;
    state.seasonDay -= 1;
    if (state.seasonDay < 1) state.seasonDay = 90;
  }

  playTone(minutes > 0 ? 560 : 440, "sine", 0.08, 0.05);
  updateClockDisplay();
}

function setSeasonDay(day) {
  state.seasonDay = Math.max(1, Math.min(90, day));
  playTone(500 + day * 3, "sine", 0.04, 0.03);
  updateClockDisplay();
}

function toggleAutoStream() {
  state.autoStream = !state.autoStream;
  const toggleBtn = document.getElementById("stream-toggle-btn");
  if (toggleBtn) {
    toggleBtn.innerText = state.autoStream ? "⏸️ Pause Stream" : "▶️ Auto-Stream";
    if (state.autoStream) {
      toggleBtn.classList.add("active-play");
    } else {
      toggleBtn.classList.remove("active-play");
    }
  }
  playTone(state.autoStream ? 680 : 420, "sine", 0.1, 0.06);
}

// Attach globally to window
window.advanceTime = advanceTime;
window.setSeasonDay = setSeasonDay;
window.toggleAutoStream = toggleAutoStream;

// Setup timeline ticker
function setupAutoStreamTicker() {
  const scrubber = document.getElementById("season-scrubber");
  if (scrubber) {
    scrubber.addEventListener("input", e => {
      setSeasonDay(parseInt(e.target.value));
    });
  }

  // Initial table & clock render
  updateClockDisplay();

  // Auto-stream interval
  setInterval(() => {
    if (state.autoStream) {
      advanceTime(5);
    }
  }, 900);
}

// --- Setup Interactive Handlers ---
function setupInteractions() {
  // Concept Tab Clicks
  document.querySelectorAll(".concept-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const concept = btn.getAttribute("data-concept");
      switchConcept(concept);
    });
  });

  // Sound Toggle
  const soundBtn = document.getElementById("sound-btn");
  if (soundBtn) {
    soundBtn.addEventListener("click", () => {
      initAudio();
      state.soundEnabled = !state.soundEnabled;
      soundBtn.innerText = state.soundEnabled ? "🔊 Sound: ON" : "🔇 Sound: OFF";
      if (state.soundEnabled) playTone(520, "sine", 0.15, 0.1);
    });
  }

  // Sliders
  const sMoist = document.getElementById("slider-moist");
  if (sMoist) {
    sMoist.addEventListener("input", e => {
      state.moisture = parseFloat(e.target.value);
      document.getElementById("label-moist").innerText = state.moisture.toFixed(1) + "%";
      runInference();
    });
  }

  const sTemp = document.getElementById("slider-temp");
  if (sTemp) {
    sTemp.addEventListener("input", e => {
      state.temperature = parseFloat(e.target.value);
      state.vpd = 0.61078 * Math.exp((17.27 * state.temperature) / (state.temperature + 237.3)) * (1.0 - state.humidity / 100.0);
      document.getElementById("label-temp").innerText = state.temperature.toFixed(1) + "°C";
      runInference();
    });
  }

  const sRain = document.getElementById("slider-rain");
  if (sRain) {
    sRain.addEventListener("input", e => {
      state.rain24h = parseFloat(e.target.value);
      document.getElementById("label-rain").innerText = state.rain24h.toFixed(1) + " mm";
      if (state.rain24h > 10) playWaterDrop();
      runInference();
    });
  }

  const checkStress = document.getElementById("check-stress");
  if (checkStress) {
    checkStress.addEventListener("change", e => {
      state.injectStress = e.target.checked;
      if (state.injectStress) playTone(300, "sawtooth", 0.25, 0.1);
      runInference();
    });
  }

  // Stress Presets
  const presets = {
    heatwave: { temp: 39.5, moist: 16.0, rain: 0.0, stress: false },
    monsoon: { temp: 23.0, moist: 38.0, rain: 32.0, stress: false },
    drought: { temp: 34.0, moist: 10.0, rain: 0.0, stress: true },
    flowering: { temp: 27.5, moist: 23.0, rain: 2.0, stress: false }
  };

  document.querySelectorAll(".preset-button").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-preset");
      const p = presets[type];
      if (!p) return;
      state.temperature = p.temp;
      state.moisture = p.moist;
      state.rain24h = p.rain;
      state.injectStress = p.stress;
      state.vpd = 0.61078 * Math.exp((17.27 * state.temperature) / (state.temperature + 237.3)) * (1.0 - state.humidity / 100.0);

      if (sTemp) sTemp.value = p.temp;
      if (sMoist) sMoist.value = p.moist;
      if (sRain) sRain.value = p.rain;
      if (checkStress) checkStress.checked = p.stress;

      document.getElementById("label-temp").innerText = p.temp.toFixed(1) + "°C";
      document.getElementById("label-moist").innerText = p.moist.toFixed(1) + "%";
      document.getElementById("label-rain").innerText = p.rain.toFixed(1) + " mm";

      playTone(600, "triangle", 0.18, 0.09);
      runInference();
    });
  });

  // Zone Pills & Tiles
  document.querySelectorAll(".zone-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const zid = parseInt(pill.getAttribute("data-zone"));
      selectZone(zid);
    });
  });

  for (let z = 1; z <= 4; z++) {
    const tile = document.getElementById(`zone-tile-${z}`);
    if (tile) {
      tile.addEventListener("click", () => {
        selectZone(z);
      });
    }
  }

  // Audit Exporter
  const dlBtn = document.getElementById("download-audit-btn");
  if (dlBtn) {
    dlBtn.addEventListener("click", () => {
      const z = zones[state.activeZone];
      const auditText = `# AgriTwin AI — Real-Time Agronomic Audit Record
Generated: ${new Date().toISOString()}
Sector: ${z.name}
Growth Stage: ${state.stage} (GDD: ${state.gdd} °C-day)

Telemetry Status:
- Root Zone Moisture: ${state.moisture.toFixed(1)}%
- Canopy Air Temp: ${state.temperature.toFixed(1)}°C
- Atmospheric VPD: ${state.vpd.toFixed(2)} kPa
- Canopy NDVI: ${state.ndvi.toFixed(2)}
- Soil pH Chemistry: ${state.ph.toFixed(1)} pH

AI Directives:
- Water Application Need: ${((Math.max(0, 28 - state.moisture)) * 2.8).toFixed(1)} mm
- Pumping Hours: ${((Math.max(0, 28 - state.moisture) * 28) / 45).toFixed(2)} hours
- Recommended Tariff Window: 04:00 AM – 06:30 AM Off-Peak
`;
      const blob = new Blob([auditText], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `AgriTwin_Field_Audit_${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
      playTone(720, "sine", 0.12, 0.08);
    });
  }
}

// Expose globally for HTML onclick handlers
window.selectZone = selectZone;
window.switchConcept = switchConcept;

// Initializer
document.addEventListener("DOMContentLoaded", () => {
  setupBgParticles();
  setupBiomeCanvas();
  setupAutoStreamTicker();
  setupInteractions();
  runInference();
});
