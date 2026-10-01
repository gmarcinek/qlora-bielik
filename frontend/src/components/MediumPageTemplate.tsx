import { PageTemplate, PageTemplateProps } from "./PageTemplate";

export function MediumPageTemplate({
  className = "",
  ...props
}: PageTemplateProps) {
  return <PageTemplate {...props} className={`medium-page ${className}`} />;
}
