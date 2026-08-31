import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchServiceGuide } from "../api";
import { renderMarkdown } from "../format";

type GuideSection = {
  id: string;
  label: string;
  level: 2 | 3;
};

function headingSlug(label: string, fallbackIndex: number): string {
  return label
    .toLocaleLowerCase("ru-RU")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "") || `section-${fallbackIndex + 1}`;
}

function prepareGuide(markdown: string): { html: string; sections: GuideSection[] } {
  if (!markdown) return { html: "", sections: [] };

  const parsed = new DOMParser().parseFromString(renderMarkdown(markdown, false), "text/html");
  const usedSlugs = new Map<string, number>();
  const headings = Array.from(parsed.body.querySelectorAll<HTMLHeadingElement>("h2, h3"));
  const sections = headings.map((heading, index): GuideSection => {
    const label = heading.textContent?.trim() || `Раздел ${index + 1}`;
    const baseSlug = headingSlug(label, index);
    const duplicateNumber = (usedSlugs.get(baseSlug) || 0) + 1;
    usedSlugs.set(baseSlug, duplicateNumber);
    const id = duplicateNumber === 1 ? baseSlug : `${baseSlug}-${duplicateNumber}`;
    heading.id = id;
    heading.tabIndex = -1;
    return { id, label, level: heading.tagName === "H2" ? 2 : 3 };
  });

  return { html: parsed.body.innerHTML, sections };
}

function TableOfContents({
  sections,
  activeId,
  onNavigate,
}: {
  sections: GuideSection[];
  activeId: string;
  onNavigate: (id: string) => void;
}) {
  return (
    <div className="help-toc-list">
      {sections.map((section) => (
        <a
          key={section.id}
          className={`help-toc-link is-level-${section.level} ${activeId === section.id ? "is-active" : ""}`}
          href={`#${section.id}`}
          data-section-id={section.id}
          aria-current={activeId === section.id ? "location" : undefined}
          onClick={(event) => {
            event.preventDefault();
            onNavigate(section.id);
          }}
        >
          {section.label}
        </a>
      ))}
    </div>
  );
}

export function HelpWorkspace({ focusHeading = "" }: { focusHeading?: string }) {
  const [guide, setGuide] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [activeId, setActiveId] = useState("");
  const workspaceRef = useRef<HTMLElement>(null);
  const articleRef = useRef<HTMLElement>(null);
  const desktopTocRef = useRef<HTMLElement>(null);
  const mobileTocRef = useRef<HTMLDetailsElement>(null);
  const renderedGuide = useMemo(() => prepareGuide(guide), [guide]);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setGuide(await fetchServiceGuide());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось загрузить справку");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const workspace = workspaceRef.current;
    const article = articleRef.current;
    if (!guide || !workspace || !article) return;

    setActiveId(renderedGuide.sections[0]?.id || "");

    let animationFrame = 0;
    const updateActiveSection = () => {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = window.requestAnimationFrame(() => {
        const compactLayout = window.matchMedia("(max-width:1100px)").matches;
        const activationLine = workspace.getBoundingClientRect().top + (compactLayout ? 104 : 52);
        let nextActiveId = renderedGuide.sections[0]?.id || "";
        const currentHeadings = article.querySelectorAll<HTMLHeadingElement>("h2, h3");
        for (const heading of currentHeadings) {
          if (heading.getBoundingClientRect().top > activationLine) break;
          nextActiveId = heading.id;
        }
        setActiveId((current) => current === nextActiveId ? current : nextActiveId);
      });
    };

    updateActiveSection();
    workspace.addEventListener("scroll", updateActiveSection, { passive: true });
    window.addEventListener("resize", updateActiveSection);
    return () => {
      window.cancelAnimationFrame(animationFrame);
      workspace.removeEventListener("scroll", updateActiveSection);
      window.removeEventListener("resize", updateActiveSection);
    };
  }, [guide, renderedGuide.sections]);

  useEffect(() => {
    const toc = desktopTocRef.current;
    const link = toc?.querySelector<HTMLElement>(`[data-section-id="${CSS.escape(activeId)}"]`);
    if (!toc || !link) return;

    const tocBounds = toc.getBoundingClientRect();
    const linkBounds = link.getBoundingClientRect();
    if (linkBounds.top < tocBounds.top + 34) {
      toc.scrollTo({ top: toc.scrollTop + linkBounds.top - tocBounds.top - 42, behavior: "smooth" });
    } else if (linkBounds.bottom > tocBounds.bottom - 10) {
      toc.scrollTo({ top: toc.scrollTop + linkBounds.bottom - tocBounds.bottom + 18, behavior: "smooth" });
    }
  }, [activeId]);

  const navigateTo = useCallback((id: string) => {
    mobileTocRef.current?.removeAttribute("open");
    const workspace = workspaceRef.current;
    const heading = articleRef.current?.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
    if (!workspace || !heading) return;

    const compactLayout = window.matchMedia("(max-width:1100px)").matches;
    const top = workspace.scrollTop
      + heading.getBoundingClientRect().top
      - workspace.getBoundingClientRect().top
      - (compactLayout ? 72 : 28);
    workspace.scrollTo({ top, behavior: "smooth" });
    setActiveId(id);
  }, []);

  useEffect(() => {
    if (!focusHeading || loading || !guide) return;
    const section = renderedGuide.sections.find((item) => item.label === focusHeading);
    if (!section) return;
    const timer = window.setTimeout(() => navigateTo(section.id), 0);
    return () => window.clearTimeout(timer);
  }, [focusHeading, loading, guide, renderedGuide.sections, navigateTo]);

  return (
    <section ref={workspaceRef} className="help-workspace" aria-label="Справка по Neo4j Assistant">
      <div className="help-guide-shell">
        {loading && <p className="help-guide-status">Загрузка справки…</p>}
        {!loading && error && (
          <div className="help-guide-error" role="alert">
            <strong>Справка недоступна</strong>
            <span>{error}</span>
            <button type="button" className="ghost-btn" onClick={() => void load()}>Повторить</button>
          </div>
        )}
        {!loading && guide && (
          <>
            <details ref={mobileTocRef} className="help-toc-mobile">
              <summary>
                <span>Содержание</span>
                <small>{renderedGuide.sections.find((section) => section.id === activeId)?.label || "Выберите раздел"}</small>
              </summary>
              <TableOfContents sections={renderedGuide.sections} activeId={activeId} onNavigate={navigateTo} />
            </details>
            <article
              ref={articleRef}
              className="help-guide md"
              dangerouslySetInnerHTML={{ __html: renderedGuide.html }}
            />
            <nav ref={desktopTocRef} className="help-toc" aria-label="Содержание справки">
              <p className="help-toc-title">Содержание</p>
              <TableOfContents sections={renderedGuide.sections} activeId={activeId} onNavigate={navigateTo} />
            </nav>
          </>
        )}
      </div>
    </section>
  );
}
