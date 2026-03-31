type EmptyArchiveProps = {
  title: string;
  description: string;
};

export function EmptyArchive({ title, description }: EmptyArchiveProps) {
  return (
    <div className="empty-archive" role="status">
      <p className="empty-archive__eyebrow">Waiting on content data</p>
      <h2 className="empty-archive__title">{title}</h2>
      <p className="empty-archive__description">{description}</p>
    </div>
  );
}
