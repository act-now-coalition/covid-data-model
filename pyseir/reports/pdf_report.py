import matplotlib; matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pprint


class PDFReport:
    """
    Generate pdf reports with text pages and plots.
    """

    def __init__(self, filename):
        self.filename = filename
        self.pdf = PdfPages(self.filename)

    def write_text_page(self, obj, title, page_heading=None, figsize=(6, 8),
                        pprint_kwargs=None, color='k', body_fontsize=4,
                        heading_fontsize=7, title_fontsize=12):
        """
        Use matplotlib to plot a blank figure and add text to it.

        Parameters
        ----------
        obj: object
            Object to print.
        title:
            Title of this section.
        page_heading: str
            Page title
        figsize: tuple
            Matplotlib figure size
        pprint_kwargs:
            passed to pprint.pformat()
        color: str
            matplotlib color
        body_fontsize: int
            Fontsize for the object printer.
        heading_fontsize: int
            Fontsize for the page heading.
        title_fontsize: int
            Fontsize for the title.
        """
        if pprint_kwargs is None:
            pprint_kwargs = {}

        if not isinstance(obj, str):
            s = pprint.pformat(obj, indent=0, **pprint_kwargs)
        else:
            s = obj
        fig = plt.figure(figsize=figsize or (7, len(s.split('\n')) * 0.4))
        if page_heading:
            plt.title(page_heading, fontsize=heading_fontsize)
        plt.text(0, .98, title, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=title_fontsize, color=color)

        plt.text(0, .9, s, transform=plt.gca().transAxes,
                 verticalalignment='top', color=color,
                 fontproperties=FontProperties(family='monospace', size=body_fontsize))
        plt.gca().axis('off')
        self.pdf.savefig(fig)
        plt.close()

    def close(self):
        """
        Close the pdf writer.  This needs to be called to trigger writing the
        PDF.
        """
        try:
            self.pdf.close()
        except AttributeError:
            pass

    def add_figure(self, fig, **kwargs):
        """
        Add a new page containing the figure.  For some seaborn objects such as
        jointplot, pass jointplot.fig.

        Parameters
        ----------
        fig: matplotlib.Figure
        kwargs: dict
            Passed to self.pdf.savefig(fig, **kwargs)
        """
        self.pdf.savefig(fig, **kwargs)
        plt.close(fig)
