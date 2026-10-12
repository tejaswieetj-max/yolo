/**
 * Context-Aware Smart Glasses - Caregiver & Researcher Web Dashboard
 *
 * Implements:
 * 1. Role-Based Login Authentication (Researcher / Caregiver / Admin)
 * 2. 180° Top-Down Spatial Radar with Kinematic Head Yaw needle
 * 3. 3-Axis Neckband LRA Pulse Visualizer (Left, Center, Right)
 * 4. Live Rolling Hazard Detection Stream Table
 * 5. Chart.js Analytics (Hazard Donut + TTC Timeline)
 * 6. 1-Click Research Dataset Export (CSV / JSON)
 * 7. Authenticated Telemetry Receiver (SSE with session token)
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM References - Auth
  const loginOverlay = document.getElementById("loginModalOverlay");
  const loginForm = document.getElementById("loginForm");
  const inputUsername = document.getElementById("inputUsername");
  const inputPassword = document.getElementById("inputPassword");
  const btnTogglePwd = document.getElementById("btnTogglePwd");
  const loginErrorMessage = document.getElementById("loginErrorMessage");
  const btnLoginSubmit = document.getElementById("btnLoginSubmit");
  const loginBtnText = document.getElementById("loginBtnText");
  const loginSpinner = document.getElementById("loginSpinner");

  const tabResearcher = document.getElementById("tabResearcher");
  const tabCaregiver = document.getElementById("tabCaregiver");
  const btnQuickResearcher = document.getElementById("btnQuickResearcher");
  const btnQuickCaregiver = document.getElementById("btnQuickCaregiver");

  const mainDashboardApp = document.getElementById("mainDashboardApp");
  const userProfileBadge = document.getElementById("userProfileBadge");
  const userAvatar = document.getElementById("userAvatar");
  const userName = document.getElementById("userName");
  const userRole = document.getElementById("userRole");
  const btnLogout = document.getElementById("btnLogout");

  // DOM References - Dashboard
  const radarCanvas = document.getElementById("spatialRadarCanvas");
  const radarCtx = radarCanvas.getContext("2d");

  const headYawDisplay = document.getElementById("headYawDisplay");
  const criticalDistVal = document.getElementById("criticalDistVal");
  const interruptStateVal = document.getElementById("interruptStateVal");
  const computeSavingsVal = document.getElementById("computeSavingsVal");
  const systemLatency = document.getElementById("systemLatency");
  const sessionTimer = document.getElementById("sessionTimer");
  const eventCountBadge = document.getElementById("eventCountBadge");
  const eventTableBody = document.getElementById("eventTableBody");
  const urgencyPill = document.getElementById("urgencyPill");

  const nodeLeft = document.getElementById("nodeLeft");
  const nodeCenter = document.getElementById("nodeCenter");
  const nodeRight = document.getElementById("nodeRight");
  const stateLeft = document.getElementById("stateLeft");
  const stateCenter = document.getElementById("stateCenter");
  const stateRight = document.getElementById("stateRight");

  const activeWaveformName = document.getElementById("activeWaveformName");
  const firingCadence = document.getElementById("firingCadence");
  const ttcVal = document.getElementById("ttcVal");

  const btnExportCsv = document.getElementById("btnExportCsv");
  const btnExportJson = document.getElementById("btnExportJson");

  // State
  let authToken = localStorage.getItem("smart_glasses_token") || null;
  let currentUser = null;
  let sseEventSource = null;
  let sessionStartTime = Date.now();
  let eventsHistory = [];
  let classCounts = { person: 0, car: 0, truck: 0, overhead_hazard: 0, generic_obstacle: 0 };
  let currentHeadYaw = 0.0;
  let currentObstacle = { dist: 2.5, azimuth: 0, class: "person", active: false };

  // ============================================================================
  // AUTHENTICATION LOGIC & SESSION CHECK
  // ============================================================================
  async function checkExistingSession() {
    if (!authToken) {
      showLoginModal();
      return;
    }

    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        handleLoginSuccess(authToken, data.user);
      } else {
        localStorage.removeItem("smart_glasses_token");
        authToken = null;
        showLoginModal();
      }
    } catch (err) {
      // If server is not responding to /api/auth/me directly, check local token
      const savedUser = localStorage.getItem("smart_glasses_user");
      if (savedUser) {
        try {
          handleLoginSuccess(authToken, JSON.parse(savedUser));
          return;
        } catch (e) {}
      }
      showLoginModal();
    }
  }

  function showLoginModal() {
    loginOverlay.classList.remove("hidden");
    mainDashboardApp.classList.add("blurred");
    if (sseEventSource) {
      sseEventSource.close();
      sseEventSource = null;
    }
  }

  function hideLoginModal() {
    loginOverlay.classList.add("hidden");
    mainDashboardApp.classList.remove("blurred");
  }

  function handleLoginSuccess(token, user) {
    authToken = token;
    currentUser = user;
    localStorage.setItem("smart_glasses_token", token);
    localStorage.setItem("smart_glasses_user", JSON.stringify(user));

    // Update Header Profile Badge
    userAvatar.textContent = user.avatar || "👤";
    userName.textContent = user.name || user.username;
    userRole.textContent = user.role || "User";

    hideLoginModal();
    connectTelemetryStream();
  }

  // Password Visibility Toggle
  btnTogglePwd.addEventListener("click", () => {
    if (inputPassword.type === "password") {
      inputPassword.type = "text";
      btnTogglePwd.textContent = "🙈";
    } else {
      inputPassword.type = "password";
      btnTogglePwd.textContent = "👁️";
    }
  });

  // Role Tab Switching
  tabResearcher.addEventListener("click", () => {
    tabResearcher.classList.add("active");
    tabCaregiver.classList.remove("active");
    inputUsername.placeholder = "Enter username or email";
  });

  tabCaregiver.addEventListener("click", () => {
    tabCaregiver.classList.add("active");
    tabResearcher.classList.remove("active");
    inputUsername.placeholder = "Enter username or email";
  });

  // Perform Login Request
  async function performLogin(username, password) {
    loginErrorMessage.style.display = "none";
    loginBtnText.textContent = "Signing In...";
    btnLoginSubmit.disabled = true;

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();
      if (res.ok && data.status === "ok") {
        handleLoginSuccess(data.token, data.user);
      } else {
        loginErrorMessage.textContent = data.detail || "Authentication failed. Check credentials.";
        loginErrorMessage.style.display = "flex";
      }
    } catch (err) {
      // Fallback offline mock authentication
      if (username === "researcher" && password === "Glasses2026!") {
        handleLoginSuccess("mock_res_token", {
          username: "researcher",
          name: "Lead Researcher",
          role: "Lead Researcher",
          avatar: "🔬",
        });
      } else if (username === "caregiver" && password === "Caregiver2026!") {
        handleLoginSuccess("mock_care_token", {
          username: "caregiver",
          name: "Primary Caregiver",
          role: "Primary Caregiver",
          avatar: "🩺",
        });
      } else {
        loginErrorMessage.textContent = "Invalid username or password.";
        loginErrorMessage.style.display = "flex";
      }
    } finally {
      loginBtnText.textContent = "Sign In to Dashboard";
      btnLoginSubmit.disabled = false;
    }
  }

  loginForm.addEventListener("submit", (e) => {
    e.preventDefault();
    performLogin(inputUsername.value, inputPassword.value);
  });

  // 1-Click Quick Demo Login Handlers
  btnQuickResearcher.addEventListener("click", () => {
    inputUsername.value = "researcher";
    inputPassword.value = "Glasses2026!";
    tabResearcher.click();
    performLogin("researcher", "Glasses2026!");
  });

  btnQuickCaregiver.addEventListener("click", () => {
    inputUsername.value = "caregiver";
    inputPassword.value = "Caregiver2026!";
    tabCaregiver.click();
    performLogin("caregiver", "Caregiver2026!");
  });

  // Logout Handler
  btnLogout.addEventListener("click", async () => {
    try {
      if (authToken) {
        await fetch("/api/auth/logout", {
          method: "POST",
          headers: { Authorization: `Bearer ${authToken}` },
        });
      }
    } catch (e) {}

    localStorage.removeItem("smart_glasses_token");
    localStorage.removeItem("smart_glasses_user");
    authToken = null;
    currentUser = null;
    showLoginModal();
  });

  // ============================================================================
  // Session Timer
  // ============================================================================
  setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const hrs = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const secs = String(elapsed % 60).padStart(2, "0");
    sessionTimer.textContent = `${hrs}:${mins}:${secs}`;
  }, 1000);

  // ============================================================================
  // Chart.js Setup
  // ============================================================================
  const ctxHazard = document.getElementById("hazardDistributionChart").getContext("2d");
  const hazardChart = new Chart(ctxHazard, {
    type: "doughnut",
    data: {
      labels: ["Pedestrian", "Vehicle", "Overhead", "Other"],
      datasets: [{
        data: [1, 1, 1, 1],
        backgroundColor: ["#10b981", "#ef4444", "#a855f7", "#3b82f6"],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: "#9ca3af", font: { size: 10 } } },
      },
      cutout: "68%",
    },
  });

  const ctxTimeline = document.getElementById("ttcTimelineChart").getContext("2d");
  const timelineChart = new Chart(ctxTimeline, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "TTC (s)",
          data: [],
          borderColor: "#00f2fe",
          backgroundColor: "rgba(0, 242, 254, 0.1)",
          tension: 0.3,
          borderWidth: 2,
          yAxisID: "yTTC",
        },
        {
          label: "Dist (m)",
          data: [],
          borderColor: "#f59e0b",
          borderDash: [4, 4],
          tension: 0.3,
          borderWidth: 1.5,
          yAxisID: "yDist",
        }
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        yTTC: {
          position: "left",
          min: 0,
          max: 6,
          ticks: { color: "#00f2fe", font: { size: 9 } },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
        yDist: {
          position: "right",
          min: 0,
          max: 4,
          ticks: { color: "#f59e0b", font: { size: 9 } },
          grid: { display: false },
        }
      },
      plugins: {
        legend: { labels: { color: "#9ca3af", font: { size: 10 } } },
      },
    },
  });

  // ============================================================================
  // 180° Spatial Radar Canvas Rendering
  // ============================================================================
  function drawRadar() {
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    radarCtx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h - 25;
    const maxRadius = h - 45;

    // Draw range arc rings (1m, 2m, 3m, 4m)
    const ranges = [1, 2, 3, 4];
    radarCtx.lineWidth = 1;
    ranges.forEach((r) => {
      const radius = (r / 4.0) * maxRadius;
      radarCtx.beginPath();
      radarCtx.arc(cx, cy, radius, Math.PI, 2 * Math.PI, false);
      radarCtx.strokeStyle = r === 2 ? "rgba(239, 68, 68, 0.4)" : "rgba(255, 255, 255, 0.08)";
      if (r === 2) {
        radarCtx.setLineDash([4, 4]); // 2.0m interrupt gate ring
      } else {
        radarCtx.setLineDash([]);
      }
      radarCtx.stroke();

      radarCtx.fillStyle = r === 2 ? "#ef4444" : "rgba(255, 255, 255, 0.3)";
      radarCtx.font = "9px JetBrains Mono";
      radarCtx.fillText(`${r}m`, cx + radius - 15, cy - 4);
    });
    radarCtx.setLineDash([]);

    // Sector dividing lines
    const anglesDeg = [-45, -15, 15, 45];
    anglesDeg.forEach((deg) => {
      const rad = (deg - 90) * (Math.PI / 180.0);
      radarCtx.beginPath();
      radarCtx.moveTo(cx, cy);
      radarCtx.lineTo(cx + Math.cos(rad) * maxRadius, cy + Math.sin(rad) * maxRadius);
      radarCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      radarCtx.stroke();
    });

    // Draw Head Yaw Compass Needle
    const yawRad = (currentHeadYaw - 90) * (Math.PI / 180.0);
    radarCtx.beginPath();
    radarCtx.moveTo(cx, cy);
    radarCtx.lineTo(cx + Math.cos(yawRad) * (maxRadius * 0.95), cy + Math.sin(yawRad) * (maxRadius * 0.95));
    radarCtx.strokeStyle = "#00f2fe";
    radarCtx.lineWidth = 2;
    radarCtx.shadowColor = "#00f2fe";
    radarCtx.shadowBlur = 8;
    radarCtx.stroke();
    radarCtx.shadowBlur = 0;

    // Torso mark
    radarCtx.beginPath();
    radarCtx.arc(cx, cy, 7, 0, 2 * Math.PI);
    radarCtx.fillStyle = "#10b981";
    radarCtx.fill();

    // Active obstacle blip
    if (currentObstacle.active && currentObstacle.dist <= 4.0) {
      const blipRad = (currentObstacle.azimuth - 90) * (Math.PI / 180.0);
      const blipDistPx = (currentObstacle.dist / 4.0) * maxRadius;
      const bx = cx + Math.cos(blipRad) * blipDistPx;
      const by = cy + Math.sin(blipRad) * blipDistPx;

      radarCtx.beginPath();
      radarCtx.arc(bx, by, 10, 0, 2 * Math.PI);
      radarCtx.fillStyle = currentObstacle.dist < 1.0 ? "rgba(239, 68, 68, 0.4)" : "rgba(245, 158, 11, 0.35)";
      radarCtx.fill();

      radarCtx.beginPath();
      radarCtx.arc(bx, by, 5, 0, 2 * Math.PI);
      radarCtx.fillStyle = currentObstacle.dist < 1.0 ? "#ef4444" : "#f59e0b";
      radarCtx.fill();

      radarCtx.fillStyle = "#ffffff";
      radarCtx.font = "10px Outfit";
      radarCtx.fillText(`${currentObstacle.dist.toFixed(1)}m (${currentObstacle.class})`, bx + 8, by - 4);
    }
  }

  // ============================================================================
  // Telemetry Packet Ingestion & UI Update
  // ============================================================================
  function handleTelemetryEvent(event) {
    eventsHistory.push(event);
    if (eventsHistory.length > 500) eventsHistory.shift();

    systemLatency.textContent = `${(event.latency_ms || 18.4).toFixed(1)} ms`;
    eventCountBadge.textContent = `${eventsHistory.length} Events`;

    // Fusion mode display
    const fusionModeEl = document.getElementById("fusionModeVal");
    const fusionDetailEl = document.getElementById("fusionDetailVal");
    if (fusionModeEl) {
      const mode = event.fusion_mode || "lite";
      fusionModeEl.textContent = mode.toUpperCase();
      fusionModeEl.style.color = mode === "full" ? "#06b6d4" : "#f59e0b";
    }
    if (fusionDetailEl) {
      const yoloMs = (event.yolo_inference_ms || 0).toFixed(1);
      const mpMs = (event.mp_inference_ms || 0).toFixed(1);
      const poseIcon = event.pose_detected ? " \u{1F9B4}" : "";
      const handsIcon = event.hands_detected ? " \u{1F91A}" : "";
      fusionDetailEl.textContent = `YOLO: ${yoloMs}ms | MP: ${mpMs}ms${poseIcon}${handsIcon}`;
    }

    currentHeadYaw = event.head_yaw_deg || 0.0;
    headYawDisplay.textContent = `${currentHeadYaw >= 0 ? "+" : ""}${currentHeadYaw.toFixed(1)}°`;

    let azAngle = 0;
    if (event.motor_channel === "LEFT") azAngle = -30;
    else if (event.motor_channel === "RIGHT") azAngle = 30;

    currentObstacle = {
      dist: event.distance_m || 4.0,
      azimuth: azAngle,
      class: event.hazard_class || "person",
      active: event.is_depth_interrupt || false,
    };
    drawRadar();

    criticalDistVal.textContent = `${(event.distance_m || 4.0).toFixed(2)} m`;
    interruptStateVal.textContent = event.is_depth_interrupt ? "TRIGGERED (<2m)" : "IDLE (>2m)";
    interruptStateVal.style.color = event.is_depth_interrupt ? "#ef4444" : "#10b981";

    const savings = event.roi_area_ratio > 0 ? (1.0 - event.roi_area_ratio) * 100 : (event.ai_gated ? 100 : 85);
    computeSavingsVal.textContent = `${savings.toFixed(1)}%`;

    activeWaveformName.textContent = `${event.waveform_name || "Strong Click - 100%"} (#${event.waveform_id || 1})`;
    ttcVal.textContent = event.ttc_seconds < 90 ? `${event.ttc_seconds.toFixed(2)} s` : "> 10 s";

    urgencyPill.textContent = event.urgency_level || "SAFE";
    urgencyPill.className = `badge-pill ${event.urgency_level ? event.urgency_level.toLowerCase() : "safe"}`;

    if (event.urgency_level === "IMMINENT") {
      firingCadence.textContent = "20 Hz (Continuous Rapid Buzz)";
    } else if (event.urgency_level === "APPROACHING") {
      firingCadence.textContent = "4.0 Hz (Double-Pulse Cadence)";
    } else {
      firingCadence.textContent = "1.0 Hz (Gentle Situational Ping)";
    }

    triggerNodePulse(event.motor_channel, event.urgency_level, event.should_pulse);
    appendTableRow(event);
    updateCharts(event);
  }

  function triggerNodePulse(channel, urgency, shouldPulse) {
    const nodes = { LEFT: nodeLeft, CENTER: nodeCenter, RIGHT: nodeRight };
    const states = { LEFT: stateLeft, CENTER: stateCenter, RIGHT: stateRight };

    ["LEFT", "CENTER", "RIGHT"].forEach((ch) => {
      nodes[ch].className = "lra-node";
      states[ch].textContent = "IDLE";
      states[ch].style.color = "var(--text-muted)";
    });

    if (channel && shouldPulse && nodes[channel]) {
      const pulseClass = urgency === "IMMINENT" ? "pulsing-red" : (urgency === "APPROACHING" ? "pulsing-yellow" : "pulsing-green");
      nodes[channel].classList.add(pulseClass);
      states[channel].textContent = urgency === "IMMINENT" ? "BUZZING!" : "PULSING";
      states[channel].style.color = urgency === "IMMINENT" ? "#ef4444" : (urgency === "APPROACHING" ? "#f59e0b" : "#10b981");

      setTimeout(() => {
        nodes[channel].classList.remove(pulseClass);
        states[channel].textContent = "IDLE";
        states[channel].style.color = "var(--text-muted)";
      }, 240);
    }
  }

  function appendTableRow(event) {
    const row = document.createElement("tr");
    const d = new Date(event.timestamp * 1000);
    const timeStr = `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`;

    const hClass = (event.hazard_class || "person").toLowerCase();
    let tagStyle = "tag-person";
    if (["car", "truck", "bus"].includes(hClass)) tagStyle = "tag-car";
    else if (hClass.includes("overhead")) tagStyle = "tag-overhead";

    row.innerHTML = `
      <td>${timeStr}</td>
      <td><span class="tag-class ${tagStyle}">${hClass}</span></td>
      <td><strong>${event.motor_channel || "CENTER"}</strong></td>
      <td>${(event.distance_m || 4.0).toFixed(2)}m</td>
      <td>${event.ttc_seconds < 90 ? event.ttc_seconds.toFixed(2) + "s" : "--"}</td>
      <td>#${event.waveform_id || 1}</td>
      <td>${event.urgency_level || "SAFE"}</td>
    `;

    eventTableBody.insertBefore(row, eventTableBody.firstChild);
    while (eventTableBody.children.length > 30) {
      eventTableBody.removeChild(eventTableBody.lastChild);
    }
  }

  function updateCharts(event) {
    const hClass = (event.hazard_class || "person").toLowerCase();
    if (["person", "pedestrian"].includes(hClass)) classCounts.person++;
    else if (["car", "bus", "truck", "motorcycle"].includes(hClass)) classCounts.car++;
    else if (hClass.includes("overhead")) classCounts.overhead_hazard++;
    else classCounts.generic_obstacle++;

    hazardChart.data.datasets[0].data = [
      classCounts.person,
      classCounts.car,
      classCounts.overhead_hazard,
      classCounts.generic_obstacle,
    ];
    hazardChart.update("none");

    if (timelineChart.data.labels.length > 20) {
      timelineChart.data.labels.shift();
      timelineChart.data.datasets[0].data.shift();
      timelineChart.data.datasets[1].data.shift();
    }
    const d = new Date(event.timestamp * 1000);
    timelineChart.data.labels.push(`${d.getSeconds()}s`);
    timelineChart.data.datasets[0].data.push(event.ttc_seconds < 10 ? event.ttc_seconds : 6);
    timelineChart.data.datasets[1].data.push(event.distance_m || 4.0);
    timelineChart.update("none");
  }

  // ============================================================================
  // 1-Click Research Dataset Export
  // ============================================================================
  btnExportCsv.addEventListener("click", () => {
    if (eventsHistory.length === 0) {
      alert("No telemetry events recorded yet.");
      return;
    }
    const headers = ["timestamp", "head_yaw_deg", "motor_channel", "distance_m", "ttc_seconds", "urgency_level", "hazard_class", "waveform_id", "waveform_name", "roi_area_ratio", "latency_ms"];
    let csvContent = "data:text/csv;charset=utf-8," + headers.join(",") + "\n";
    eventsHistory.forEach((e) => {
      const row = [
        e.timestamp,
        e.head_yaw_deg,
        e.motor_channel,
        e.distance_m,
        e.ttc_seconds,
        e.urgency_level,
        e.hazard_class,
        e.waveform_id,
        `"${e.waveform_name}"`,
        e.roi_area_ratio,
        e.latency_ms,
      ];
      csvContent += row.join(",") + "\n";
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `smart_glasses_trials_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  btnExportJson.addEventListener("click", () => {
    if (eventsHistory.length === 0) {
      alert("No telemetry events recorded yet.");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(eventsHistory, null, 2));
    const link = document.createElement("a");
    link.setAttribute("href", dataStr);
    link.setAttribute("download", `smart_glasses_session_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // ============================================================================
  // Authenticated Telemetry Receiver
  // ============================================================================
  function connectTelemetryStream() {
    if (!authToken) return;

    if (sseEventSource) {
      sseEventSource.close();
      sseEventSource = null;
    }

    try {
      sseEventSource = new EventSource(`/api/stream?token=${encodeURIComponent(authToken)}`);
      sseEventSource.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          handleTelemetryEvent(data);
        } catch (err) {}
      };
      sseEventSource.onerror = () => {
        sseEventSource.close();
        // Fallback to active mock simulation if backend stream is quiet
        startFallbackSimulationLoop();
      };
    } catch (e) {
      startFallbackSimulationLoop();
    }
  }

  function startFallbackSimulationLoop() {
    console.log("[Dashboard] Fallback simulation loop disabled. Waiting for real webcam data.");
  }

  // Initial setup: draw radar & check auth session
  drawRadar();
  checkExistingSession();
});
