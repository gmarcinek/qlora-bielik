import { ReactNode } from "react";
import { PageLayout } from "./PageLayout";

export type PageTemplateProps = {
  eyebrow: string;
  title: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function PageTemplate({
  eyebrow,
  title,
  actions,
  children,
  className,
}: PageTemplateProps) {
  return (
    <PageLayout className={className}>
      <div className="d-flex flex-wrap justify-content-between gap-3 mb-1">
        <div>
          <p className="section-kicker">{eyebrow}</p>
          <h1>{title}</h1>
        </div>
        {actions}
      </div>
      {children}
    </PageLayout>
  );
}
