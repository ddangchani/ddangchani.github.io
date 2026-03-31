import Script from "next/script";

type AnalyticsProps = {
  trackingId?: string;
};

export function Analytics({ trackingId }: AnalyticsProps) {
  if (!trackingId) {
    return null;
  }

  return (
    <>
      <Script src={`https://www.googletagmanager.com/gtag/js?id=${trackingId}`} strategy="afterInteractive" />
      <Script id="ga-tracker" strategy="afterInteractive">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', '${trackingId}');
        `}
      </Script>
    </>
  );
}
