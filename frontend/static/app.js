// Drives the 3-step flow: submit a set name -> enter a pack count -> show
// results. No framework, just fetch() calls against the two backend JSON
// endpoints and direct DOM updates — matches the "deliberately simple"
// frontend scope.

const setForm = document.getElementById("set-form");
const setInput = document.getElementById("set-input");
const setMessage = document.getElementById("set-message");
const stepY = document.getElementById("step-y");
const chosenSetLabel = document.getElementById("chosen-set");
const yForm = document.getElementById("y-form");
const yInput = document.getElementById("y-input");
const yMessage = document.getElementById("y-message");
const stepResults = document.getElementById("step-results");
const resultsStatus = document.getElementById("results-status");
const resultsContent = document.getElementById("results-content");
const histogramImg = document.getElementById("histogram");

let confirmedSetName = null;

setForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const setname = setInput.value.trim();
    setMessage.textContent = "Checking...";
    setMessage.className = "message";

    const response = await fetch("/validate_set", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ setname }),
    });
    const data = await response.json();

    if (data.valid) {
        confirmedSetName = data.setname;
        setMessage.textContent = "Set recognized.";
        setMessage.className = "message success";
        chosenSetLabel.textContent = confirmedSetName;
        stepY.hidden = false;
        // A different set invalidates any previously-shown results.
        stepResults.hidden = true;
        resultsContent.hidden = true;
        yMessage.textContent = "";
    } else {
        confirmedSetName = null;
        setMessage.textContent = data.message;
        setMessage.className = "message error";
        stepY.hidden = true;
    }
});

yForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!confirmedSetName) return;

    const min = parseInt(yInput.min, 10);
    const max = parseInt(yInput.max, 10);
    const y = parseInt(yInput.value, 10);

    // Client-side check for immediate feedback; the server enforces the
    // same range authoritatively regardless, since this can be bypassed.
    if (!Number.isInteger(y) || y < min || y > max) {
        yMessage.textContent = `Enter a whole number between ${min} and ${max}.`;
        yMessage.className = "message error";
        return;
    }
    yMessage.textContent = "";

    stepResults.hidden = false;
    resultsContent.hidden = true;
    resultsStatus.textContent = `Simulating ${y} pack${y !== 1 ? "s" : ""}...`;
    resultsStatus.className = "message";

    const response = await fetch("/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ setname: confirmedSetName, y }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        resultsStatus.textContent = errorData.error || "Simulation failed.";
        resultsStatus.className = "message error";
        return;
    }

    const data = await response.json();
    resultsStatus.textContent = "";
    resultsContent.hidden = false;

    const money = (value) => `$${value.toFixed(2)}`;
    // VaR is a loss magnitude; showing it as a percentage of total_cost
    // (what was actually spent on the y packs) makes it directly readable
    // at a glance ("this is 49% of what I paid") instead of requiring the
    // reader to mentally divide the two numbers themselves.
    const varLabel = (value, binding) => {
        const pctOfCost = data.total_cost > 0 ? ` (${((value / data.total_cost) * 100).toFixed(0)}% of total cost)` : "";
        return binding ? `${money(value)}${pctOfCost}` : `${money(value)} (non-binding)`;
    };

    document.getElementById("m-n-trials").textContent = data.n_trials.toLocaleString();
    document.getElementById("m-pack-cost").textContent = money(data.pack_cost);
    document.getElementById("m-total-cost").textContent = money(data.total_cost);
    document.getElementById("m-mean").textContent = money(data.mean);
    document.getElementById("m-sd").textContent = money(data.sd);
    document.getElementById("m-breakeven").textContent = `${(data.breakeven_prob * 100).toFixed(2)}%`;
    document.getElementById("m-var95").textContent = varLabel(data.var_95, data.var_95_binding);
    document.getElementById("m-cvar95").textContent = money(data.cvar_95);
    document.getElementById("m-var99").textContent = varLabel(data.var_99, data.var_99_binding);
    document.getElementById("m-cvar99").textContent = money(data.cvar_99);

    // Top-tier: the odds (and payoff) of landing in the rare upper tail,
    // estimated via importance sampling rather than the plain Monte Carlo
    // batch the other rows come from — see ARCHITECTURE.md section 1.8 for
    // why this needs a different technique than VaR/CVaR do.
    const tailPct = data.top_tier_tail_pct.toFixed(2);
    document.getElementById("m-top-tier-label").textContent = `Odds of a top-${tailPct}% pull`;
    document.getElementById("m-top-tier-prob").textContent = `${(data.top_tier_probability * 100).toFixed(3)}%`;
    document.getElementById("m-top-tier-ev").textContent = money(data.top_tier_expected_profit);
    const essPct = ((data.top_tier_ess / data.top_tier_n_trials) * 100).toFixed(1);
    document.getElementById("m-top-tier-ess").textContent =
        `effective sample size: ${Math.round(data.top_tier_ess).toLocaleString()} / ${data.top_tier_n_trials.toLocaleString()} trials (${essPct}%)`;

    histogramImg.src = `data:image/png;base64,${data.histogram_base64}`;
});
