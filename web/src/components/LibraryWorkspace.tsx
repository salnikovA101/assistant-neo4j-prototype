import { IconLibrary } from "./Icons";

export function LibraryWorkspace() {
  return (
    <section className="article-library" aria-labelledby="article-library-title">
      <div className="article-library-inner">
        <header className="article-library-header">
          <div>
            <p className="article-library-kicker">ИСТОЧНИКИ ЗНАНИЙ</p>
            <h1 id="article-library-title">Библиотека статей</h1>
            <p>Научные публикации и документы, из которых формируется база знаний.</p>
          </div>
          <span className="article-library-status">В разработке</span>
        </header>

        <div className="article-library-placeholder">
          <div className="article-library-icon" aria-hidden="true">
            <IconLibrary />
          </div>
          <h2>Библиотека пока пуста</h2>
          <p>
            Здесь появится единое хранилище статей с метаданными, поиском и связями с сущностями базы.
          </p>
          <div className="article-library-plan" aria-label="Запланированные возможности">
            <span>Поиск по статьям</span>
            <span>Метаданные и источники</span>
            <span>Связь с базой</span>
          </div>
        </div>
      </div>
    </section>
  );
}
