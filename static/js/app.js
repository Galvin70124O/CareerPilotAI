/* CareerPilot AI - page behaviour (no framework).
   1. Theme toggle        4. Loading state      7. What-If Simulator
   2. Tabs                5. Microphone         8. Results bar ("Edit skills")
   3. Drop zone           6. Scroll to results                                  */
(function () {
  "use strict";

  var MAX_BYTES = 2 * 1024 * 1024;
  var ALLOWED = ["pdf", "docx", "txt"];
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (selector, root) { return (root || document).querySelector(selector); };
  var $$ = function (selector, root) { return Array.prototype.slice.call((root || document).querySelectorAll(selector)); };

  /* ---------- 1. Theme toggle (remembered in localStorage) ---------- */
  function setTheme(theme, remember) {
    document.documentElement.setAttribute("data-theme", theme);
    var toggle = $("#theme-toggle");
    if (toggle) {
      toggle.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    }
    if (remember) {
      try { localStorage.setItem("theme", theme); } catch (e) { /* private mode: just don't remember */ }
    }
  }

  function initTheme() {
    var toggle = $("#theme-toggle");
    if (!toggle) return;
    setTheme(document.documentElement.getAttribute("data-theme") || "light", false);
    toggle.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      setTheme(next, true);
    });
    // Follow the system setting until the user picks a theme themselves
    var media = window.matchMedia("(prefers-color-scheme: dark)");
    var onSystemChange = function (event) {
      var saved = null;
      try { saved = localStorage.getItem("theme"); } catch (e) {}
      if (!saved) setTheme(event.matches ? "dark" : "light", false);
    };
    if (media.addEventListener) media.addEventListener("change", onSystemChange);
  }

  /* ---------- 2. Tabs (ARIA tabs: click, arrow keys, Home/End) ---------- */
  var selectTab = function () {};   // set by initTabs, also used by "Edit skills"

  function initTabs() {
    var tabs = $$('[role="tab"]');
    if (!tabs.length) return;

    function select(tab, focus) {
      tabs.forEach(function (other) {
        var selected = other === tab;
        other.setAttribute("aria-selected", selected ? "true" : "false");
        other.tabIndex = selected ? 0 : -1;
        document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
      });
      if (focus) tab.focus();
    }
    selectTab = select;

    tabs.forEach(function (tab, index) {
      tab.addEventListener("click", function () { select(tab, false); });
      tab.addEventListener("keydown", function (event) {
        var next = null;
        if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
        if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
        if (event.key === "Home") next = tabs[0];
        if (event.key === "End") next = tabs[tabs.length - 1];
        if (next) { event.preventDefault(); select(next, true); }
      });
    });
  }

  /* ---------- 3. Drop zone: click or drop a file, then analyze right away ---------- */
  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function checkFile(file) {
    var extension = file.name.split(".").pop().toLowerCase();
    if (ALLOWED.indexOf(extension) === -1) return "Please choose a PDF, DOCX or TXT file.";
    if (file.size > MAX_BYTES) return "This file is " + formatSize(file.size) + ". The limit is 2 MB.";
    return "";
  }

  function initDropzone() {
    var zone = $("[data-dropzone]");
    var input = $("#resume_file");
    var form = $("#analyze-form");
    var errorBox = $("#upload-error");
    if (!zone || !input || !form) return;

    function useFile(file) {
      var problem = checkFile(file);
      errorBox.textContent = problem;
      if (problem) {
        zone.classList.remove("has-file");
        input.value = "";
        return;
      }
      $("[data-file-name]", zone).textContent = file.name;
      $("[data-file-size]", zone).textContent = formatSize(file.size) + " · analyzing…";
      zone.classList.add("has-file");
      $("#input_text").value = "";   // the server uses the file when there is one
      startLoading("Reading " + file.name + " and comparing it with 42 careers");
      form.submit();                 // one step: picking a file analyzes it
    }

    input.addEventListener("change", function () {
      if (input.files && input.files[0]) useFile(input.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.add("is-dragover");
      });
    });
    ["dragleave", "dragend", "drop"].forEach(function (name) {
      zone.addEventListener(name, function () { zone.classList.remove("is-dragover"); });
    });
    zone.addEventListener("drop", function (event) {
      event.preventDefault();
      var files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;
      try { input.files = files; } catch (e) { /* very old browsers: the file can't be attached */ }
      useFile(files[0]);
    });
  }

  /* ---------- 4. Loading state: skeleton dashboard + disabled buttons ---------- */
  function startLoading(message) {
    var loading = $("#loading");
    var results = $("#results");
    var preview = $(".preview");
    if (results) results.hidden = true;
    if (preview) preview.hidden = true;
    if (loading) {
      if (message) $("[data-loading-status]", loading).textContent = message;
      loading.hidden = false;
      loading.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
    var form = $("#analyze-form");
    if (form) form.setAttribute("aria-busy", "true");
    $$("#analyze-form button[type=submit]").forEach(function (button) {
      button.disabled = true;
      button.classList.add("is-loading");
      var label = $("[data-button-label]", button);
      if (label) label.textContent = "Analyzing…";
    });
  }

  function initForm() {
    var form = $("#analyze-form");
    if (!form) return;
    form.addEventListener("submit", function (event) {
      var textPanel = $("#panel-text");
      var text = $("#input_text").value.trim();
      // Typing tab: don't send an empty form
      if (textPanel && !textPanel.hidden && !text) {
        event.preventDefault();
        $("#input_text").focus();
        return;
      }
      startLoading(textPanel && !textPanel.hidden ? "Comparing your skills with 42 careers" : null);
    });
    // Coming back with the Back button: undo the loading state
    window.addEventListener("pageshow", function (event) {
      if (!event.persisted) return;
      $("#loading").hidden = true;
      $$("#analyze-form button[type=submit]").forEach(function (button) {
        button.disabled = false;
        button.classList.remove("is-loading");
        var label = $("[data-button-label]", button);
        if (label) label.textContent = "Analyze skills";
      });
    });
  }

  /* ---------- 5. Microphone: speech to text (Chrome / Edge) ---------- */
  function initMic() {
    var button = $("#mic-button");
    var status = $("#mic-status");
    var textarea = $("#input_text");
    if (!button) return;
    var Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      button.disabled = true;
      button.title = "Speech input works in Chrome and Edge";
      return;
    }
    var recognition = null;

    function stop(message) {
      button.setAttribute("aria-pressed", "false");
      $("[data-mic-label]", button).textContent = "Speak";
      status.textContent = message || "";
    }

    button.addEventListener("click", function () {
      if (button.getAttribute("aria-pressed") === "true") {
        if (recognition) recognition.stop();
        stop();
        return;
      }
      recognition = new Recognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onresult = function (event) {
        var spoken = event.results[0][0].transcript;
        textarea.value += (textarea.value.trim() ? ", " : "") + spoken;
        stop("Added: " + spoken);
      };
      recognition.onerror = function (event) {
        stop(event.error === "not-allowed" ? "Microphone access was blocked." : "Didn't catch that, try again.");
      };
      recognition.onend = function () {
        if (button.getAttribute("aria-pressed") === "true") stop();
      };
      button.setAttribute("aria-pressed", "true");
      $("[data-mic-label]", button).textContent = "Stop";
      status.textContent = "Listening…";
      recognition.start();
    });
  }

  /* ---------- 6. Scroll to results after an analysis ---------- */
  function initResults() {
    var results = $("#results");
    if (!results) return;
    results.focus({ preventScroll: true });   // screen readers start at the results
    results.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }

  /* ---------- 7. What-If Simulator: POST /api/simulate, show before -> after ---------- */
  function deltaBadge(before, after, unit) {
    var change = after - before;
    var badge = document.createElement("span");
    if (change === 0) {
      badge.className = "badge";
      badge.textContent = "no change";
    } else {
      badge.className = "badge " + (change > 0 ? "badge-up" : "badge-down");
      badge.textContent = (change > 0 ? "+" : "−") + Math.abs(change) + unit;
    }
    return badge;
  }

  function cell(row, text) {
    var td = row.insertCell();
    td.textContent = text;
    return td;
  }

  function afterCell(row, text, badge) {
    var td = row.insertCell();
    var wrap = document.createElement("span");
    wrap.className = "sim-after";
    var value = document.createElement("span");
    value.textContent = text;
    wrap.appendChild(value);
    if (badge) wrap.appendChild(badge);
    td.appendChild(wrap);
  }

  // Built with textContent (never innerHTML), so nothing from the server is treated as HTML
  function showSimulation(result) {
    var before = result.before, after = result.after;
    var table = document.createElement("table");
    table.className = "sim-table";
    var caption = table.createCaption();
    caption.className = "sr-only";
    caption.textContent = "Before and after adding " + result.added_skill_labels.join(", ");
    var head = table.createTHead().insertRow();
    ["", "Before", "After"].forEach(function (text) {
      var th = document.createElement("th");
      th.scope = "col";
      th.textContent = text;
      head.appendChild(th);
    });
    var body = table.createTBody();

    var row = body.insertRow();
    cell(row, "Top career");
    cell(row, before.top_careers[0].label);
    var changed = before.top_careers[0].label !== after.top_careers[0].label;
    var careerBadge = null;
    if (changed) { careerBadge = document.createElement("span"); careerBadge.className = "badge badge-accent"; careerBadge.textContent = "new"; }
    afterCell(row, after.top_careers[0].label, careerBadge);

    row = body.insertRow();
    cell(row, "Confidence");
    cell(row, before.top_careers[0].percent + "%");
    afterCell(row, after.top_careers[0].percent + "%",
              changed ? null : deltaBadge(before.top_careers[0].percent, after.top_careers[0].percent, " pts"));

    row = body.insertRow();
    cell(row, "Top 3");
    var list = function (careers) { return careers.map(function (c) { return c.label + " " + c.percent + "%"; }).join(", "); };
    cell(row, list(before.top_careers));
    afterCell(row, list(after.top_careers), null);

    row = body.insertRow();
    cell(row, "Skills covered (" + result.career_label + ")");
    cell(row, before.skill_gap_percent + "%");
    afterCell(row, after.skill_gap_percent + "%", deltaBadge(before.skill_gap_percent, after.skill_gap_percent, " pts"));

    if (before.resume_score !== null) {
      row = body.insertRow();
      cell(row, "Resume score");
      cell(row, before.resume_score + " / 100");
      afterCell(row, after.resume_score + " / 100", deltaBadge(before.resume_score, after.resume_score, ""));
    }
    $("#cp-sim-result").replaceChildren(table);
  }

  function initSimulator() {
    var button = $("#cp-sim-button");
    var dataTag = $("#cp-sim-data");
    if (!button || !dataTag) return;
    var data = JSON.parse(dataTag.textContent);
    var errorBox = $("#cp-sim-error");
    var label = $("[data-button-label]", button);

    button.addEventListener("click", function () {
      var skills = $$("#sim-options input:checked").map(function (box) { return box.value; });
      errorBox.textContent = "";
      if (!skills.length) {
        errorBox.textContent = "Pick at least one skill first.";
        return;
      }
      button.disabled = true;
      button.classList.add("is-loading");
      label.textContent = "Simulating…";

      fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: data.text, career: data.career, skills: skills, is_resume: data.is_resume })
      })
        .then(function (response) {
          return response.json().then(function (body) {
            if (!response.ok) throw new Error(body.error || "Something went wrong.");
            return body;
          });
        })
        .then(showSimulation)
        .catch(function (error) { errorBox.textContent = error.message; })
        .finally(function () {
          button.disabled = false;
          button.classList.remove("is-loading");
          label.textContent = "Simulate";
        });
    });
  }

  /* ---------- 8. Results bar: "Edit skills" opens the input card above the results ---------- */
  function initEditSkills() {
    var button = $("#edit-skills");
    var panel = $("#analyze-panel");
    if (!button || !panel) return;
    button.addEventListener("click", function () {
      var opening = panel.hidden;
      panel.hidden = !opening;
      button.setAttribute("aria-expanded", opening ? "true" : "false");
      if (opening) {
        selectTab($("#tab-text"), false);
        var textarea = $("#input_text");
        textarea.focus({ preventScroll: true });
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
        panel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    initTabs();
    initDropzone();
    initForm();
    initMic();
    initSimulator();
    initEditSkills();
    initResults();
  });
})();
