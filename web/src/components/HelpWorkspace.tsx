import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchServiceGuide } from "../api";
import { renderMarkdown } from "../format";

type GuideSection = {
  id: string;
  label: string;
  level: 2 | 3;
  searchText: string;
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
    const chunks = [label];
    const stop = headings[index + 1];
    for (let node = heading.nextSibling; node && node !== stop; node = node.nextSibling) {
      const text = node.textContent?.trim();
      if (text) chunks.push(text);
    }
    return {
      id,
      label,
      level: heading.tagName === "H2" ? 2 : 3,
      searchText: chunks.join(" ").toLocaleLowerCase("ru-RU"),
    };
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
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const workspaceRef = useRef<HTMLElement>(null);
  const articleRef = useRef<HTMLElement>(null);
  const desktopTocRef = useRef<HTMLElement>(null);
  const mobileTocRef = useRef<HTMLDetailsElement>(null);
  const searchWrapRef = useRef<HTMLDivElement>(null);
  const renderedGuide = useMemo(() => prepareGuide(guide), [guide]);
  const searchResults = useMemo(() => {
    const query = searchQuery.trim().toLocaleLowerCase("ru-RU");
    if (!query) return [];
    return renderedGuide.sections.filter((section) => section.searchText.includes(query)).slice(0, 8);
  }, [renderedGuide.sections, searchQuery]);
  const quickLinks = [
    { label: "Начать работу", heading: "Быстрый старт" },
    { label: "Выбрать режим", heading: "Режимы работы" },
    { label: "Проверить источники", heading: "Как читать ответ" },
    { label: "Работать с графом", heading: "Вся база: ручное исследование графа" },
    { label: "Заполнить карточку", heading: "Карточки" },
  ];

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setGuide(await fetchServiceGuide());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось загрузить помощь");
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

  useEffect(() => {
    if (!searchOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSearchOpen(false);
    };
    const onPointer = (event: PointerEvent) => {
      if (!searchWrapRef.current?.contains(event.target as Node)) setSearchOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onPointer);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onPointer);
    };
  }, [searchOpen]);

  return (
    <section ref={workspaceRef} className="help-workspace" aria-label="Помощь по Neo4j Assistant">
      <div className="help-guide-shell">
        {loading && <p className="help-guide-status">Загрузка помощи…</p>}
        {!loading && error && (
          <div className="help-guide-error" role="alert">
            <strong>Помощь недоступна</strong>
            <span>{error}</span>
            <button type="button" className="ghost-btn" onClick={() => void load()}>Повторить</button>
          </div>
        )}
        {!loading && guide && (
          <>
            <header className="help-hero">
              <div><h1>Помощь</h1><p>Найдите нужное действие или перейдите к частому сценарию.</p></div>
              <div className="help-search-wrap" ref={searchWrapRef}>
                <input
                  type="search"
                  value={searchQuery}
                  onChange={(event) => { setSearchQuery(event.target.value); setSearchOpen(true); }}
                  onFocus={() => { if (searchQuery.trim()) setSearchOpen(true); }}
                  placeholder="Поиск по руководству"
                  aria-label="Поиск по руководству"
                />
                {searchOpen && searchQuery && <div className="help-search-results">
                  {searchResults.map((section) => <button key={section.id} type="button" onClick={() => { navigateTo(section.id); setSearchQuery(""); setSearchOpen(false); }}>{section.label}</button>)}
                  {searchResults.length === 0 && <p>Раздел не найден</p>}
                </div>}
              </div>
              <nav className="help-quick-links" aria-label="Частые сценарии">
                {quickLinks.map((item) => {
                  const section = renderedGuide.sections.find((entry) => entry.label === item.heading);
                  return <button key={item.label} type="button" disabled={!section} onClick={() => section && navigateTo(section.id)}>{item.label}</button>;
                })}
              </nav>
            </header>
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
            <nav ref={desktopTocRef} className="help-toc" aria-label="Содержание помощи">
              <p className="help-toc-title">Содержание</p>
              <TableOfContents sections={renderedGuide.sections} activeId={activeId} onNavigate={navigateTo} />
            </nav>
          </>
        )}
      </div>
    </section>
  );
}
