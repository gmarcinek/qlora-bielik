import { ReactNode } from "react";

type PageLayoutProps = {
  children: ReactNode;
  className?: string;
};

export function PageLayout({ children, className = "" }: PageLayoutProps) {
  return (
    <section className={`workspace-page ${className}`}>{children}</section>
  );
}
