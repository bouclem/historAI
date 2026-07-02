/* ============================================
   historAI — Website JavaScript
   Handles: .md loading, timeline filtering,
   basic syntax highlighting
   ============================================ */

/* --- Load a markdown file and render its code block --- */
async function loadCodeBlock(mdPath, targetId) {
    const container = document.getElementById(targetId);
    if (!container) return;

    try {
        const response = await fetch(mdPath);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const markdown = await response.text();

        // Extract code from ```python ... ``` blocks
        const codeMatch = markdown.match(/```(?:python|py)?\n([\s\S]*?)```/);
        const rawCode = codeMatch ? codeMatch[1] : markdown;

        // Apply basic syntax highlighting
        const highlighted = highlightPython(rawCode);

        // Build the code block
        const pre = document.createElement("pre");
        pre.innerHTML = highlighted;
        container.querySelector(".code-content").appendChild(pre);
    } catch (err) {
        container.querySelector(".code-content").innerHTML =
            `<p style="color: var(--text-muted); font-family: var(--font-mono); font-size: 0.85rem;">// Failed to load source code: ${err.message}</p>`;
    }
}

/* --- Minimal Python syntax highlighting --- */
function highlightPython(code) {
    // Escape HTML first
    let html = code
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Comments (triple-quoted strings and # comments)
    html = html.replace(/(#[^\n]*)/g, '<span class="cmt">$1</span>');

    // Triple-quoted strings
    html = html.replace(/("""[\s\S]*?""")/g, '<span class="str">$1</span>');

    // Single/double quoted strings (not inside tags)
    html = html.replace(/(?<!>)("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="str">$1</span>');

    // Keywords
    const keywords = [
        "def", "class", "import", "from", "return", "if", "else", "elif",
        "for", "while", "in", "not", "and", "or", "is", "None", "True",
        "False", "self", "try", "except", "raise", "with", "as", "pass",
        "break", "continue", "lambda", "yield", "global", "nonlocal",
        "assert", "del", "async", "await"
    ];
    const kwPattern = new RegExp(`\\b(${keywords.join("|")})\\b`, "g");
    html = html.replace(kwPattern, '<span class="kw">$1</span>');

    // Numbers
    html = html.replace(/\b(\d+\.?\d*)\b/g, '<span class="num">$1</span>');

    // Function calls — word followed by (
    html = html.replace(/\b([a-z_][a-z_0-9]*)\(/gi, '<span class="fn">$1</span>(');

    return html;
}

/* --- Timeline filtering --- */
function initTimelineFilter() {
    const filterBtns = document.querySelectorAll(".filter-btn");
    const entries = document.querySelectorAll(".timeline-entry");

    if (filterBtns.length === 0) return;

    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const filter = btn.dataset.filter;

            // Update active state
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Filter entries
            entries.forEach(entry => {
                if (filter === "all" || entry.classList.contains(filter)) {
                    entry.style.display = "";
                } else {
                    entry.style.display = "none";
                }
            });
        });
    });
}

/* --- Init on page load --- */
document.addEventListener("DOMContentLoaded", () => {
    initTimelineFilter();

    // Auto-load any code blocks with data-md-source
    document.querySelectorAll("[data-md-source]").forEach(el => {
        const mdPath = el.dataset.mdSource;
        const targetId = el.id;
        loadCodeBlock(mdPath, targetId);
    });
});
