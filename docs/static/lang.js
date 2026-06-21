let standLang = localStorage.getItem("standLang") || "es";

function toggleStandLang() {
  standLang = standLang === "es" ? "en" : "es";
  try { localStorage.setItem("standLang", standLang); } catch(e){}
  applyStandLang();
}

function applyStandLang() {
  document.querySelectorAll("[data-es]").forEach(el => {
    el.textContent = standLang === "es" ? el.dataset.es : el.dataset.en;
  });
  document.querySelectorAll("[data-es-html]").forEach(el => {
    el.innerHTML = standLang === "es" ? el.dataset.esHtml : el.dataset.enHtml;
  });
  const btn = document.getElementById("langToggle");
  if (btn) btn.textContent = standLang === "es" ? "EN" : "ES";
}

function injectLangButton() {
  const btn = document.createElement("button");
  btn.id = "langToggle";
  btn.textContent = standLang === "es" ? "EN" : "ES";
  btn.onclick = toggleStandLang;
  Object.assign(btn.style, {
    position: "fixed", top: "12px", right: "12px", zIndex: "9999",
    background: "rgba(255,255,255,0.12)", border: "1px solid rgba(255,255,255,0.2)",
    color: "#fff", padding: "6px 16px", borderRadius: "8px", fontSize: "12px",
    fontWeight: "700", cursor: "pointer", letterSpacing: "1px", backdropFilter: "blur(8px)",
  });
  document.body.appendChild(btn);
}

document.addEventListener("DOMContentLoaded", () => {
  injectLangButton();
  applyStandLang();
});
