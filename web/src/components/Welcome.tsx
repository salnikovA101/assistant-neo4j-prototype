export function Welcome({ onHelp }: { onHelp: () => void }) {
  return <div className="welcome welcome-minimal"><button type="button" className="welcome-help-link" onClick={onHelp}>Как пользоваться</button></div>;
}
