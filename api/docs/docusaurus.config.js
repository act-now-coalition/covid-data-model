module.exports = {
  title: "Covid Act Now",
  tagline: "API Documentation",
  url: "https://api.covidactnow.org",
  baseUrl: "/",
  onBrokenLinks: "throw",
  favicon: "img/favicon.ico",
  organizationName: "facebook", // Usually your GitHub org/user name.
  projectName: "docusaurus", // Usually your repo name.
  themeConfig: {
    navbar: {
      title: "",
      logo: {
        alt: "Covid Act Now Logo",
        src: "img/can_logo.png",
      },
      items: [
        
        {
          to: "/",
          activeBasePath: "/",
          label: "Guide",
          position: "left",
        },
        {
          to: "api",
          activeBasePath: "api",
          label: "API Reference",
          position: "left",
        },

        {
          href: "https://github.com/covid-projections/covid-data-model",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "API Reference",
              to: "docs/",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "Facebook",
              href: "https://www.facebook.com/covidactnow",
            },
            {
              label: "Instagram",
              href: "https://www.instagram.com/covidactnow",
            },
            {
              label: "Twitter",
              href: "https://twitter.com/CovidActNow",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "Blog",
              href: "https://blog.covidactnow.org/",
            },
            {
              label: "Covid Act Now",
              href: "https://covidactnow.org",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Covid Act Now. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          routeBasePath: "/",

          // Please change this to your repo.
          
        },

        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
};
