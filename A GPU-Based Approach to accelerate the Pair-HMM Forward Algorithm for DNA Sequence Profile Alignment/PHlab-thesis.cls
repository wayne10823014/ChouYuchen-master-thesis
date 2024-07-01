\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{PHlab-thesis}[2021/09/08 v1.0 NCKU Masters/PhD Thesis]
\LoadClass[12pt,a4paper,openany]{book}

\RequirePackage{etoolbox}

\RequirePackage{fonts-config}

\RequirePackage{array}

\RequirePackage{amsmath}
\RequirePackage{unicode-math}

\RequirePackage{booktabs}%  For \{top|mid|bottom}rule
\RequirePackage{fancyhdr}
\RequirePackage{tabularx}%  Sometimes more convenient than \tabular or \tabular*

\RequirePackage{geometry}
\RequirePackage{type1cm}
\geometry{verbose,a4paper,tmargin=2.3cm,bmargin=3cm,lmargin=2.5cm,rmargin=3cm}

\RequirePackage{setspace}
\onehalfspace

\hyphenpenalty=1024

\RequirePackage[bottom]{footmisc}

\RequirePackage[intoc]{nomencl}
\makenomenclature



\DeclareOption{PhD}{\newcommand*\PhD{}}

\newcommand*\BibLaTeXSortingSpec{nty}
\DeclareOption{BibLaTeXsortingNone}{\renewcommand\BibLaTeXSortingSpec{none}}

\DeclareOption*{\ClassError{Phlab-thesis class passed unknown option: '\@ptionlist{\@currname.\@currext}'}{Nothing more to say.}}
\ProcessOptions\relax

%──────────  configure biblatex & hyperref  ──────────
\ifdefined\optHyperlinks
\RequirePackage[hyperfootnotes=false]{hyperref}
\hypersetup{colorlinks,citecolor=blue,filecolor=blue,linkcolor=blue,urlcolor=blue}
\RequirePackage[sorting=\BibLaTeXSortingSpec]{biblatex}
\else
\RequirePackage[draft,hyperfootnotes=false]{hyperref}
\RequirePackage[sorting=\BibLaTeXSortingSpec,hyperref=false]{biblatex}
\fi

\renewbibmacro{in:}{}


\RequirePackage{graphicx}

\ifdefined\optWatermark
\RequirePackage{eso-pic} %provides \AddToShipoutPicture
\fi




%\newcommand*\Chap[1]{\chapter*{#1}\AddToContents{#1}}
\newcommand*\退[1]{\hspace{-#1pt}}% Useful to manually adjust space between characters


\newcommand*\學位{\ifdef{\PhD}{碩士}{碩士}}

\newcommand*\University中文{國立成功大學}
\newcommand*\University英文{National Cheng Kung University}

\renewcommand{\listfigurename}{List of Figures}
\renewcommand{\listtablename}{List of Tables}
\renewcommand{\bibname}{References}



\ifdefined\optWatermark
\AddToShipoutPicture{%
\put(0,0){%
\parbox[b][\paperheight]{\paperwidth}{\vfill\centering%
\includegraphics[]{NCKU-watermark-A4.pdf}%
\vfill}}}
\fi
