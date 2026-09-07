// Drives the 3-step flow: submit a set name -> pick a y -> show results.
// No framework, just fetch() calls against the two backend JSON endpoints
// and direct DOM updates — matches the "deliberately simple" frontend scope.

const setForm = document.getElementById("set-form");
const setInput = document.getElementById("set-input");
const setMessage = document.getElementById("set-message");
const stepY = document.getElementById("step-y");
const chosenSetLabel = document.getElementById("chosen-set");
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
    } else {
        confirmedSetName = null;
        setMessage.textContent = data.message;
        setMessage.className = "message error";
        stepY.hidden = true;
    }
});

document.getElementById("y-buttons").addEventListener("click", async (event) => {
    if (!event.target.classList.contains("y-button")) return;
    if (!confirmedSetName) return;

    document.querySelectorAll(".y-button").forEach((btn) => btn.classList.remove("selected"));
    event.target.classList.add("selected");

    const y = parseInt(event.target.dataset.y, 10);

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
    const varLabel = (value, binding) => (binding ? money(value) : `${money(value)} (non-binding)`);

    document.getElementById("m-n-trials").textContent = data.n_trials.toLocaleString();
    document.getElementById("m-pack-cost").textContent = money(data.pack_cost);
    document.getElementById("m-mean").textContent = money(data.mean);
    document.getElementById("m-sd").textContent = money(data.sd);
    document.getElementById("m-breakeven").textContent = `${(data.breakeven_prob * 100).toFixed(2)}%`;
    document.getElementById("m-var95").textContent = varLabel(data.var_95, data.var_95_binding);
    document.getElementById("m-cvar95").textContent = money(data.cvar_95);
    document.getElementById("m-var99").textContent = varLabel(data.var_99, data.var_99_binding);
    document.getElementById("m-cvar99").textContent = money(data.cvar_99);

    histogramImg.src = `data:image/png;base64,${data.histogram_base64}`;
});
