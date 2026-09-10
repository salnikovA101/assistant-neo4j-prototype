import { useState } from "react";
import { IconLibrary } from "./Icons";

const sections = [
  {
    id: "articles",
    label: "Научные статьи",
    description: "Здесь будут научные публикации, на основе которых создана база знаний.",
    features: ["Поиск по статьям", "Метаданные и источники", "Связь с базой"],
  },
  {
    id: "regulations",
    label: "Нормативные документы",
    description: "Здесь будут нормативные документы для проверки заквасок и продуктов. Проверка ассистентом — в разработке.",
    features: ["Поиск по нормативам", "Проверка карточек", "Использование ассистентом"],
  },
];

export function LibraryWorkspace() {
  const [activeTab, setActiveTab] = useState(sections[0].id);
  return (
    <section className="article-library" aria-labelledby="article-library-title">
      <div className="article-library-inner">
        <header className="article-library-header">
          <div>
            <p className="article-library-kicker">ИСТОЧНИКИ ЗНАНИЙ</p>
            <h1 id="article-library-title">Библиотека документов</h1>
            <p>Научные статьи и нормативные документы.</p>
          </div>
          <span className="article-library-status">В разработке</span>
        </header>

        <div className="document-library-tabs" role="tablist" aria-label="Виды документов">
          {sections.map((section, index) => (
            <button
              key={section.id}
              type="button"
              role="tab"
              id={`document-tab-${section.id}`}
              aria-controls={`document-panel-${section.id}`}
              aria-selected={activeTab === section.id}
              tabIndex={activeTab === section.id ? 0 : -1}
              onClick={() => setActiveTab(section.id)}
              onKeyDown={(event) => {
                let nextIndex = index;
                if (event.key === "ArrowRight") nextIndex = (index + 1) % sections.length;
                else if (event.key === "ArrowLeft") nextIndex = (index + sections.length - 1) % sections.length;
                else if (event.key === "Home") nextIndex = 0;
                else if (event.key === "End") nextIndex = sections.length - 1;
                else return;
                event.preventDefault();
                setActiveTab(sections[nextIndex].id);
                document.getElementById(`document-tab-${sections[nextIndex].id}`)?.focus();
              }}
            >{section.label}</button>
          ))}
        </div>
        {sections.map((section) => (
          <div key={section.id} role="tabpanel" id={`document-panel-${section.id}`} aria-labelledby={`document-tab-${section.id}`} hidden={activeTab !== section.id} tabIndex={0}>
            <div className="article-library-placeholder">
              <div className="article-library-icon" aria-hidden="true"><IconLibrary /></div>
              <h2>{section.label} · Скоро</h2>
              <p>{section.description}</p>
              <div className="article-library-plan" aria-label="Запланированные возможности">
                {section.features.map((feature) => <span key={feature}>{feature}</span>)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
