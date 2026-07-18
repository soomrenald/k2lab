import { app } from "/scripts/app.js";

const COLORS = ["#ef5350", "#42a5f5", "#66bb6a", "#ffca28", "#ab6ce3", "#26c6ba", "#ec6fab", "#8d9eff"];

function defaultProject() {
    return {
        schema: "k2-region-lab-comfy",
        version: 1,
        canvas: { width: 1024, height: 1024 },
        global_prompt: "",
        global_negative: "",
        regions: [],
        emphases: [],
        regional: {
            enabled: true,
            inside_strength: 1.0,
            outside_penalty: 1.0,
            feather_pixels: 128,
            subject_competition: true,
            subject_fill: true,
            relaxation: true,
            late_step_scale: 0.35,
            lora_delta_adaptation: false,
            lora_delta_adaptation_gain: 0.35,
        },
        projector: {
            enabled: false,
            preset: "filter_bypass2",
            values: [0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, 0, 0],
            multiplier: 1.0,
            identity_protection: 1.0,
        },
        lora_assignments: {},
    };
}

function normalizeProject(raw) {
    const fallback = defaultProject();
    let value;
    try { value = typeof raw === "string" ? JSON.parse(raw) : structuredClone(raw); }
    catch (_) { value = fallback; }
    if (value?.schema === "k2-region-lab-project") {
        const generation = value.generation || {};
        value = {
            ...fallback,
            canvas: value.canvas || fallback.canvas,
            global_prompt: generation.global_prompt || "",
            global_negative: generation.global_negative || "",
            regions: value.regions || [],
            emphases: generation.prompt_emphases || [],
            regional: {
                enabled: generation.regional_prompting ?? true,
                inside_strength: generation.regional_prompt_strength ?? 1,
                outside_penalty: generation.regional_outside_penalty ?? 1,
                feather_pixels: generation.regional_feather_pixels ?? 128,
                subject_competition: generation.regional_subject_competition ?? true,
                subject_fill: generation.regional_subject_fill ?? true,
                relaxation: generation.regional_relaxation ?? true,
                late_step_scale: generation.regional_late_step_scale ?? 0.35,
                lora_delta_adaptation: generation.regional_lora_delta_adaptation ?? false,
                lora_delta_adaptation_gain: generation.regional_lora_delta_adaptation_gain ?? 0.35,
            },
            projector: {
                enabled: generation.projector_enabled ?? false,
                preset: generation.projector_preset || "filter_bypass2",
                values: generation.projector_values || fallback.projector.values,
                multiplier: generation.projector_multiplier ?? 1,
                identity_protection: generation.projector_identity_protection ?? 1,
            },
        };
    }
    value = { ...fallback, ...(value || {}) };
    value.canvas = { ...fallback.canvas, ...(value.canvas || {}) };
    value.regional = { ...fallback.regional, ...(value.regional || {}) };
    value.projector = { ...fallback.projector, ...(value.projector || {}) };
    value.regions = Array.isArray(value.regions) ? value.regions : [];
    value.emphases = Array.isArray(value.emphases) ? value.emphases : [];
    value.lora_assignments = value.lora_assignments || {};
    return value;
}

function element(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key === "checked") node.checked = Boolean(value);
        else if (key === "value") node.value = value;
        else node.setAttribute(key, value);
    }
    for (const child of children) node.append(child);
    return node;
}

function input(type, value, onChange, attrs = {}) {
    const field = element("input", { type, value, ...attrs });
    if (type === "checkbox") field.checked = Boolean(value);
    field.addEventListener(type === "checkbox" ? "change" : "input", () => {
        let next = field.value;
        if (type === "checkbox") next = field.checked;
        else if (type === "number" || type === "range") next = Number(next);
        onChange(next);
    });
    return field;
}

function textarea(value, onChange, rows = 3) {
    const field = element("textarea", { rows });
    field.value = value || "";
    field.addEventListener("input", () => onChange(field.value));
    return field;
}

function select(options, value, onChange, multiple = false) {
    const field = element("select", multiple ? { multiple: "multiple" } : {});
    for (const option of options) {
        const pair = Array.isArray(option) ? option : [option, option];
        const item = element("option", { value: pair[0], text: pair[1] });
        if (multiple ? (value || []).includes(pair[0]) : pair[0] === value) item.selected = true;
        field.append(item);
    }
    field.addEventListener("change", () => onChange(
        multiple ? [...field.selectedOptions].map(item => item.value) : field.value
    ));
    return field;
}

function row(label, control) {
    return element("label", { class: "k2-field" }, [element("span", { text: label }), control]);
}

function installStyles() {
    if (document.getElementById("k2-region-lab-styles")) return;
    const style = element("style", { id: "k2-region-lab-styles" });
    style.textContent = `
      .k2-app{font:12px system-ui;color:var(--fg-color,#ddd);background:#17191d;border:1px solid #414650;border-radius:6px;overflow:hidden;height:100%;box-sizing:border-box;user-select:none}
      .k2-app *{box-sizing:border-box}.k2-tabs{display:flex;background:#22262c;border-bottom:1px solid #414650;position:sticky;top:0;z-index:2}
      .k2-tabs button{flex:1;border:0;border-right:1px solid #414650;background:transparent;color:#bbb;padding:7px 4px;cursor:pointer}.k2-tabs button.active{background:#39404a;color:#fff}
      .k2-panel{display:none;height:calc(100% - 31px);overflow:auto;padding:9px}.k2-panel.active{display:block}
      .k2-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.k2-field{display:flex;flex-direction:column;gap:3px;margin-bottom:7px}.k2-field>span{color:#aeb5c0}
      .k2-app input,.k2-app textarea,.k2-app select,.k2-app button{font:inherit}.k2-app input,.k2-app textarea,.k2-app select{width:100%;color:#eee;background:#252930;border:1px solid #4a515d;border-radius:4px;padding:5px}
      .k2-app input[type=checkbox]{width:auto}.k2-app textarea{resize:vertical;user-select:text}.k2-app button{color:#eee;background:#343a44;border:1px solid #555e6d;border-radius:4px;padding:5px 8px;cursor:pointer}.k2-app button:hover{background:#414955}
      .k2-canvas-wrap{position:relative;background:#0e1013;border:1px solid #4a515d;border-radius:4px;overflow:hidden;margin:5px 0}.k2-canvas{display:block;width:100%;height:auto;aspect-ratio:1/1;touch-action:none;cursor:crosshair}
      .k2-columns{display:grid;grid-template-columns:minmax(260px,1.3fr) minmax(210px,1fr);gap:9px}.k2-region-list{max-height:150px;overflow:auto;border:1px solid #424955;border-radius:4px}
      .k2-region-row{display:grid;grid-template-columns:14px 1fr auto auto;align-items:center;gap:4px;padding:4px;border-bottom:1px solid #353a43;cursor:pointer}.k2-region-row.selected{background:#394252}.k2-swatch{height:11px;border-radius:2px}
      .k2-actions{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}.k2-section{font-weight:650;color:#e5e7eb;margin:8px 0 6px;border-bottom:1px solid #3d434d;padding-bottom:3px}.k2-note{color:#929aa6;font-size:11px;margin:4px 0 8px}
      .k2-emphasis{display:grid;grid-template-columns:1fr 1.5fr .7fr auto;gap:5px;align-items:center;margin-bottom:5px}.k2-vector{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}.k2-json{font-family:ui-monospace,monospace;font-size:10px}
      @media(max-width:700px){.k2-columns,.k2-grid{grid-template-columns:1fr}}
    `;
    document.head.append(style);
}

function createRegionLab(node, projectWidget, fullApp) {
    installStyles();
    let project = normalizeProject(projectWidget.value);
    let selected = project.regions.length ? 0 : -1;
    let dragStart = null;
    let draftBox = null;
    const root = element("div", { class: "k2-app" });
    const tabBar = element("div", { class: "k2-tabs" });
    const bodyPanels = new Map();
    root.append(tabBar);

    const sync = () => {
        projectWidget.value = JSON.stringify(project);
        projectWidget.callback?.(projectWidget.value);
        node.setDirtyCanvas?.(true, true);
    };

    const addTab = (name, label) => {
        const button = element("button", { text: label });
        const panel = element("div", { class: "k2-panel" });
        tabBar.append(button);
        root.append(panel);
        bodyPanels.set(name, { button, panel });
        button.addEventListener("click", event => {
            event.stopPropagation();
            for (const value of bodyPanels.values()) {
                value.button.classList.remove("active");
                value.panel.classList.remove("active");
            }
            button.classList.add("active");
            panel.classList.add("active");
        });
        return panel;
    };

    const regionsPanel = addTab("regions", "Regions");
    const spatialPanel = fullApp ? addTab("spatial", "Spatial") : null;
    const emphasisPanel = fullApp ? addTab("emphasis", "Emphasis") : null;
    const projectorPanel = fullApp ? addTab("projector", "Projector") : null;
    const loraPanel = fullApp ? addTab("loras", "LoRA scope") : null;
    bodyPanels.get("regions").button.click();

    const globalFields = element("div");
    if (fullApp) {
        globalFields.append(
            row("Global prompt", textarea(project.global_prompt, value => { project.global_prompt = value; sync(); }, 3)),
            row("Global negative", textarea(project.global_negative, value => { project.global_negative = value; sync(); }, 2)),
        );
    }
    const dimensions = element("div", { class: "k2-grid" }, [
        row("Width", input("number", project.canvas.width, value => { project.canvas.width = Math.max(16, value); sync(); draw(); }, { min: 16, max: 16384, step: 16 })),
        row("Height", input("number", project.canvas.height, value => { project.canvas.height = Math.max(16, value); sync(); draw(); }, { min: 16, max: 16384, step: 16 })),
    ]);
    const canvas = element("canvas", { class: "k2-canvas", width: 720, height: 720 });
    const canvasWrap = element("div", { class: "k2-canvas-wrap" }, [canvas]);
    const list = element("div", { class: "k2-region-list" });
    const editor = element("div");
    const columns = element("div", { class: "k2-columns" }, [element("div", {}, [canvasWrap, list]), editor]);
    regionsPanel.append(globalFields, dimensions, element("div", { class: "k2-note", text: "Drag on the canvas to create a region. Click a box or row to edit it; rows are front-to-back." }), columns);

    function point(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(project.canvas.width, (event.clientX - rect.left) / rect.width * project.canvas.width)),
            y: Math.max(0, Math.min(project.canvas.height, (event.clientY - rect.top) / rect.height * project.canvas.height)),
        };
    }

    function hitTest(position) {
        for (let index = 0; index < project.regions.length; index++) {
            const box = project.regions[index].box;
            if (position.x >= box.x0 && position.x <= box.x1 && position.y >= box.y0 && position.y <= box.y1) return index;
        }
        return -1;
    }

    canvas.addEventListener("pointerdown", event => {
        event.stopPropagation();
        canvas.setPointerCapture(event.pointerId);
        const position = point(event);
        const hit = hitTest(position);
        if (hit >= 0 && !event.shiftKey) {
            selected = hit;
            draftBox = null;
            refresh();
            return;
        }
        dragStart = position;
        draftBox = { x0: position.x, y0: position.y, x1: position.x, y1: position.y };
    });
    canvas.addEventListener("pointermove", event => {
        if (!dragStart) return;
        event.stopPropagation();
        const position = point(event);
        draftBox = {
            x0: Math.min(dragStart.x, position.x), y0: Math.min(dragStart.y, position.y),
            x1: Math.max(dragStart.x, position.x), y1: Math.max(dragStart.y, position.y),
        };
        draw();
    });
    canvas.addEventListener("pointerup", event => {
        if (!dragStart) return;
        event.stopPropagation();
        if (draftBox && draftBox.x1 - draftBox.x0 >= 16 && draftBox.y1 - draftBox.y0 >= 16) {
            const number = project.regions.length + 1;
            project.regions.push({
                id: `region-${Date.now()}-${number}`,
                name: `Region ${number}`,
                box: Object.fromEntries(Object.entries(draftBox).map(([key, value]) => [key, Math.round(value)])),
                prompt: "", negative_prompt: "", face_identity_prompt: "",
                enabled: true, priority: number, spatial_role: "auto",
            });
            selected = project.regions.length - 1;
            sync();
        }
        dragStart = null;
        draftBox = null;
        refresh();
    });

    function drawBox(ctx, box, color, label, active, dashed = false) {
        const sx = canvas.width / project.canvas.width;
        const sy = canvas.height / project.canvas.height;
        const x = box.x0 * sx, y = box.y0 * sy, w = (box.x1 - box.x0) * sx, h = (box.y1 - box.y0) * sy;
        ctx.save();
        if (dashed) ctx.setLineDash([8, 5]);
        ctx.fillStyle = `${color}30`;
        ctx.strokeStyle = color;
        ctx.lineWidth = active ? 4 : 2;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        if (label) {
            ctx.font = "bold 15px system-ui";
            const width = ctx.measureText(label).width + 10;
            ctx.fillStyle = color;
            ctx.fillRect(x, Math.max(0, y - 22), width, 22);
            ctx.fillStyle = "#101216";
            ctx.fillText(label, x + 5, Math.max(16, y - 6));
        }
        ctx.restore();
    }

    function draw() {
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#101318";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#242a33";
        ctx.lineWidth = 1;
        for (let index = 1; index < 8; index++) {
            const x = index * canvas.width / 8;
            const y = index * canvas.height / 8;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }
        project.regions.slice().reverse().forEach((region, reverseIndex) => {
            const index = project.regions.length - 1 - reverseIndex;
            if (region.enabled !== false) drawBox(ctx, region.box, COLORS[index % COLORS.length], region.name, index === selected);
        });
        if (draftBox) drawBox(ctx, draftBox, "#fff", "New region", false, true);
    }

    function renderList() {
        list.replaceChildren();
        project.regions.forEach((region, index) => {
            const regionRow = element("div", { class: `k2-region-row ${index === selected ? "selected" : ""}` }, [
                element("span", { class: "k2-swatch", style: `background:${COLORS[index % COLORS.length]}` }),
                element("span", { text: region.name }),
            ]);
            const up = element("button", { text: "↑", title: "Move toward front" });
            const down = element("button", { text: "↓", title: "Move toward back" });
            up.addEventListener("click", event => { event.stopPropagation(); if (index > 0) { [project.regions[index - 1], project.regions[index]] = [project.regions[index], project.regions[index - 1]]; selected = index - 1; sync(); refresh(); } });
            down.addEventListener("click", event => { event.stopPropagation(); if (index < project.regions.length - 1) { [project.regions[index + 1], project.regions[index]] = [project.regions[index], project.regions[index + 1]]; selected = index + 1; sync(); refresh(); } });
            regionRow.append(up, down);
            regionRow.addEventListener("click", () => { selected = index; refresh(); });
            list.append(regionRow);
        });
    }

    function regionEditor() {
        editor.replaceChildren();
        if (selected < 0 || !project.regions[selected]) {
            editor.append(element("div", { class: "k2-note", text: "Draw or select a region to edit its label and prompts." }));
            return;
        }
        const region = project.regions[selected];
        const update = (key, value) => { region[key] = value; sync(); draw(); renderList(); };
        const geometry = element("div", { class: "k2-grid" });
        for (const key of ["x0", "y0", "x1", "y1"]) geometry.append(row(key.toUpperCase(), input("number", Math.round(region.box[key]), value => { region.box[key] = Math.round(value); sync(); draw(); }, { step: 1 })));
        const remove = element("button", { text: "Delete region" });
        remove.addEventListener("click", () => { project.regions.splice(selected, 1); selected = Math.min(selected, project.regions.length - 1); sync(); refresh(); });
        editor.append(
            row("Region name", input("text", region.name, value => update("name", value))),
            element("div", { class: "k2-grid" }, [
                row("Spatial role", select([["auto", "Auto"], ["subject", "Subject target"], ["background", "Background band"]], region.spatial_role || "auto", value => update("spatial_role", value))),
                row("Priority", input("number", region.priority ?? 0, value => update("priority", Math.round(value)), { step: 1 })),
            ]),
            row("Enabled", input("checkbox", region.enabled !== false, value => update("enabled", value))),
            row("Face identity prompt", textarea(region.face_identity_prompt, value => update("face_identity_prompt", value), 3)),
            row("Regional prompt", textarea(region.prompt, value => update("prompt", value), 4)),
            row("Regional negative prompt", textarea(region.negative_prompt, value => update("negative_prompt", value), 3)),
            element("div", { class: "k2-section", text: "Geometry" }), geometry,
            remove,
        );
    }

    function refresh() { draw(); renderList(); regionEditor(); renderLoraAssignments(); }

    if (spatialPanel) {
        const addSetting = (label, key, type, attrs = {}) => spatialPanel.append(row(label, input(type, project.regional[key], value => { project.regional[key] = value; sync(); }, attrs)));
        addSetting("Use unified spatial prompting", "enabled", "checkbox");
        const spatialGrid = element("div", { class: "k2-grid" });
        spatialPanel.append(spatialGrid);
        const addGrid = (label, key, attrs) => spatialGrid.append(row(label, input("number", project.regional[key], value => { project.regional[key] = value; sync(); }, attrs)));
        addGrid("Inside boost", "inside_strength", { min: 0.1, max: 10, step: 0.1 });
        addGrid("Outside penalty", "outside_penalty", { min: 0, max: 10, step: 0.1 });
        addGrid("Spatial falloff (px)", "feather_pixels", { min: 0, max: 2048, step: 16 });
        addGrid("Late-step spatial scale", "late_step_scale", { min: 0, max: 1, step: 0.05 });
        addGrid("LoRA delta response", "lora_delta_adaptation_gain", { min: 0, max: 1, step: 0.05 });
        for (const [label, key] of [["Exclusive overlapping subjects", "subject_competition"], ["Make subjects fill their boxes", "subject_fill"], ["Relax guidance during late steps", "relaxation"], ["Adapt guidance from regional LoRA delta", "lora_delta_adaptation"]]) addSetting(label, key, "checkbox");
    }

    if (emphasisPanel) {
        emphasisPanel.append(element("div", { class: "k2-note", text: "Phrase boosts become native ComfyUI weighted-prompt syntax before CLIP encoding. Scope can be Global or any named region." }));
        const emphasisList = element("div");
        const renderEmphases = () => {
            emphasisList.replaceChildren();
            project.emphases.forEach((item, index) => {
                const scopeOptions = [["global", "Global"], ...project.regions.map(region => [region.id, region.name])];
                const remove = element("button", { text: "×" });
                remove.addEventListener("click", () => { project.emphases.splice(index, 1); sync(); renderEmphases(); });
                const itemRow = element("div", { class: "k2-emphasis" }, [
                    select(scopeOptions, item.scope_id || "global", value => { item.scope_id = value; sync(); }),
                    input("text", item.phrase || "", value => { item.phrase = value; sync(); }),
                    input("number", item.strength ?? 0.5, value => { item.strength = value; sync(); }, { min: 0, max: 2, step: 0.05 }),
                    remove,
                ]);
                emphasisList.append(itemRow);
            });
        };
        const add = element("button", { text: "Add phrase emphasis" });
        add.addEventListener("click", () => { project.emphases.push({ scope_id: "global", phrase: "", strength: 0.5, occurrence: 0 }); sync(); renderEmphases(); });
        emphasisPanel.append(emphasisList, add);
        renderEmphases();
    }

    if (projectorPanel) {
        projectorPanel.append(element("div", { class: "k2-note", text: "Applies Krea 2 txtfusion.projector weights before regional LoRA routing." }));
        projectorPanel.append(row("Apply global projector vector", input("checkbox", project.projector.enabled, value => { project.projector.enabled = value; sync(); })));
        const presets = {
            filter_bypass2: [0,0,0,0,0,0,0,0,-0.5117,-0.8906,0,0],
            filter_bypass3: [0,0,0,0,0,0,0,0,-0.5117,-0.8906,-0.6094,0],
            skc3vo: [-5.44,-16.11,-37.11,-50.39,-70.7,-39.45,-39.84,-143.7511,-51.17,-89.06,-60.94,-11.28],
            z0jglf: [-13.6,-40.275,-92.775,-159.75,-176.75,-98.625,-99.6,-359.3778,-127.925,-222.65,-152.35,-28.2],
        };
        const vector = element("div", { class: "k2-vector" });
        const renderVector = () => {
            vector.replaceChildren();
            project.projector.values.forEach((value, index) => vector.append(row(`V${index + 1}`, input("number", value, next => { project.projector.values[index] = next; project.projector.preset = "custom"; sync(); }, { step: 0.0001 }))));
        };
        projectorPanel.append(row("Preset", select([...Object.keys(presets), "custom"], project.projector.preset, value => { project.projector.preset = value; if (presets[value]) project.projector.values = [...presets[value]]; sync(); renderVector(); })), vector,
            element("div", { class: "k2-grid" }, [
                row("Global multiplier", input("number", project.projector.multiplier, value => { project.projector.multiplier = value; sync(); }, { min: -20, max: 20, step: 0.1 })),
                row("Face identity protection", input("number", project.projector.identity_protection, value => { project.projector.identity_protection = value; sync(); }, { min: 0, max: 1, step: 0.05 })),
            ]));
        renderVector();
    }

    let loraAssignmentContainer = null;
    if (loraPanel) {
        loraPanel.append(element("div", { class: "k2-note", text: "Connect K2 LoRA Reference nodes to LoRA slots 1–4, then assign each slot to one or more named regions here. Use native Load LoRA for global LoRAs." }));
        loraAssignmentContainer = element("div");
        loraPanel.append(loraAssignmentContainer);
    }
    function renderLoraAssignments() {
        if (!loraAssignmentContainer) return;
        loraAssignmentContainer.replaceChildren();
        for (let slot = 1; slot <= 4; slot++) {
            const key = String(slot);
            loraAssignmentContainer.append(row(`LoRA slot ${slot} targets`, select(project.regions.map(region => [region.name, region.name]), project.lora_assignments[key] || [], value => { project.lora_assignments[key] = value; sync(); }, true)));
        }
    }

    if (fullApp) {
        const details = element("details");
        const summary = element("summary", { text: "Project JSON import/export" });
        const jsonArea = element("textarea", { class: "k2-json", rows: 8 });
        jsonArea.value = JSON.stringify(project, null, 2);
        const load = element("button", { text: "Load JSON" });
        const refreshJson = element("button", { text: "Refresh export" });
        load.addEventListener("click", () => { project = normalizeProject(jsonArea.value); selected = project.regions.length ? 0 : -1; sync(); refresh(); });
        refreshJson.addEventListener("click", () => { jsonArea.value = JSON.stringify(project, null, 2); });
        details.append(summary, jsonArea, element("div", { class: "k2-actions" }, [load, refreshJson]));
        regionsPanel.append(details);
    }

    root.addEventListener("pointerdown", event => event.stopPropagation());
    root.addEventListener("wheel", event => event.stopPropagation(), { passive: true });
    refresh();
    sync();
    return root;
}

app.registerExtension({
    name: "k2.region.lab.app",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["K2RegionEditor", "K2RegionLabApp"].includes(nodeData.name)) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const projectWidget = this.widgets?.find(widget => widget.name === "project_json");
            if (!projectWidget) return result;
            projectWidget.type = "converted-widget";
            projectWidget.computeSize = () => [0, -4];
            const fullApp = nodeData.name === "K2RegionLabApp";
            const editor = createRegionLab(this, projectWidget, fullApp);
            this.addDOMWidget("k2_region_ui", "k2-region-ui", editor, {
                serialize: false,
                hideOnZoom: false,
                getValue: () => undefined,
                setValue: () => {},
            });
            this.setSize(fullApp ? [880, 920] : [720, 720]);
            return result;
        };
    },
});
