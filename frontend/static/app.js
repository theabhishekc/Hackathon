let META = null;
let FORM_STATE = {};

function el(tag, attrs={}, children=[]) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k,v]) => {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  });
  children.forEach(c => e.appendChild(c));
  return e;
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  const data = await res.json();
  if (!data.ok) { alert("Meta error: " + data.error); return; }
  META = data.meta;
  document.getElementById("metaInfo").innerHTML = `
    <div><b>Features:</b> ${META.features.join(", ")}</div>
    <div><b>Numeric:</b> ${META.numeric_features.join(", ")}</div>
    <div><b>Categorical:</b> ${META.categorical_features.join(", ")}</div>
    <div><b>AUC:</b> ${META.auc ? META.auc.toFixed(3) : "N/A"}</div>
  `;
  // Populate sample selector
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = "";
  META.train_sample.forEach((row, idx) => {
    const lbl = Object.values(row).slice(0,3).join(" | ");
    const opt = el("option", { value: idx, text: `#${idx+1} – ${lbl}` });
    sel.appendChild(opt);
  });
  renderForm();
}

function renderForm(row=null) {
  const area = document.getElementById("formArea");
  area.innerHTML = "";
  const grid = el("div", { class: "grid" });
  META.features.forEach(f => {
    let v = row ? row[f] : (FORM_STATE[f] ?? "");
    const isNum = META.numeric_features.includes(f);
    const input = el("input", { type: isNum ? "number" : "text", value: v ?? "" });
    input.addEventListener("input", () => FORM_STATE[f] = input.value);
    const wrap = el("div", {}, [
      el("label", { text: f }),
      input
    ]);
    grid.appendChild(wrap);
  });
  area.appendChild(grid);
}

async function predict() {
  const method = document.getElementById("rateMethod").value;
  // Build applicant dict from FORM_STATE
  const applicant = {};
  META.features.forEach(f => {
    let v = FORM_STATE[f];
    if (v === undefined || v === "") { applicant[f] = null; return; }
    if (META.numeric_features.includes(f)) {
      applicant[f] = Number(v);
    } else {
      applicant[f] = v; // keep string
    }
  });
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ applicant, rate_method: method })
  });
  const data = await res.json();
  if (!data.ok) { alert("Predict error: " + data.error); console.log(data.trace); return; }

  document.getElementById("pdVal").textContent = (data.pd*100).toFixed(2) + "%";
  document.getElementById("rateVal").textContent = (data.interest_rate*100).toFixed(2) + "%";
  document.getElementById("amountVal").textContent = "₹ " + Math.round(data.loan_amount_used).toLocaleString("en-IN");
  document.getElementById("profitVal").textContent = "₹ " + Math.round(data.expected_profit).toLocaleString("en-IN");

  const items = data.explanation.top_contributions;
  const labels = items.map(it => it.feature);
  const values = items.map(it => it.contribution);
  renderBarChart("contribChart", labels, values);

  // basic NL explanation
  const positives = items.filter(x => x.contribution > 0).slice(0,3).map(x=>x.feature);
  const negatives = items.filter(x => x.contribution < 0).slice(0,3).map(x=>x.feature);
  const why = `Risk is driven up by ${positives.join(", ") || "—"} and mitigated by ${negatives.join(", ") || "—"}.`;
  const summary = `Predicted default probability is ${(data.pd*100).toFixed(1)}%. Assigned rate ${(data.interest_rate*100).toFixed(2)}% balances expected interest income versus potential loss (LGD 60%). ${why}`;
  document.getElementById("nlExplanation").textContent = summary;
}

let CHARTS = {};
function renderBarChart(canvasId, labels, values) {
  if (CHARTS[canvasId]) {
    CHARTS[canvasId].destroy();
  }
  const ctx = document.getElementById(canvasId).getContext("2d");
  CHARTS[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Log-odds contribution (wᵢ·xᵢ)",
        data: values,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: { enabled: true }
      },
      scales: {
        x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 0 } },
        y: { beginAtZero: false }
      }
    }
  });
}

async function runPortfolio() {
  const res = await fetch("/api/portfolio_dataset");
  const data = await res.json();
  if (!data.ok) { alert("Portfolio error: " + data.error); return; }
  document.getElementById("accCount").textContent = data.accepted_count;
  document.getElementById("totCount").textContent = data.total_applicants;
  document.getElementById("totProfit").textContent = "₹ " + Math.round(data.total_profit_top30).toLocaleString("en-IN");

  renderLineChart("cumProfitChart", data.cum_profit_x, data.cum_profit_y);
}

function renderLineChart(canvasId, xs, ys) {
  if (CHARTS[canvasId]) CHARTS[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext("2d");
  CHARTS[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels: xs,
      datasets: [{ label: "Cumulative expected profit", data: ys, fill: false }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } },
      scales: { x: { title: { display: true, text: "Applicants (sorted by EP)" } },
                y: { title: { display: true, text: "Cumulative Profit (₹)" } } }
    }
  });
}

document.getElementById("refreshMeta").addEventListener("click", loadMeta);
document.getElementById("loadSample").addEventListener("click", () => {
  const idx = parseInt(document.getElementById("sampleSelect").value, 10);
  const row = META.train_sample[idx];
  FORM_STATE = { ...row };
  renderForm(row);
});
document.getElementById("predictBtn").addEventListener("click", predict);
document.getElementById("runPortfolio").addEventListener("click", runPortfolio);

// auto load meta on first paint
window.addEventListener("load", loadMeta);
