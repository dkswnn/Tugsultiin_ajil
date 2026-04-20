% Copied/classic thesis class from previous years
\NeedsTeXFormat{LaTeX2e}[2017/04/15]
\ProvidesClass{Thesis}[2022/04/12 v0.1 testclass]
% Pass any options (e.g., 11pt,a4paper) through to the base class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass{report}

% --- contents below mirrors your template/Thesis.cls (trimmed minimally)
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{graphicx}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{longtable}
\usepackage[titles]{tocloft}
\usepackage{hyperref}
\usepackage{etoolbox}
% default indent (overridden below to 6pt per requirement)
\setlength{\parindent}{15pt}
\setlength{\parskip}{0pt}
\usepackage{vmargin}
% Margins per requirement: left 25mm (2.5cm), right 20mm (2.0cm), top 20mm, bottom 20mm
\setmarginsrb {25mm}{20mm}{20mm}{20mm}{20pt}{0.25in}{9pt}{0.3in}
% Single spacing per requirement
\setstretch{1.0}
% First-line paragraph indent: 6pt per requirement
\setlength{\parindent}{6pt}
\AtBeginDocument{%
	\fussy
	\leftskip=0pt
	\rightskip=0pt
	\parfillskip=0pt plus 1fil
	\pretolerance=1000
	\tolerance=3000
	\emergencystretch=2em
}
\hypersetup{urlcolor=black, colorlinks=true, linkcolor=black, citecolor=black}

\pagestyle{fancy}
\fancyhf{}%
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\normalsize \thepage}%

% Heading spacing and formats (titlesec)
\titlespacing*{\chapter}{0pt}{0pt}{6pt}
\titleformat{\chapter}[block]{\bfseries}{\thechapter.}{0.5em}{}
\titlespacing{\section}{0pt}{0pt}{4pt}
\titleformat{\section}[block]{\bfseries}{\thesection.}{0.5em}{}
\titlespacing{\subsection}{0pt}{0pt}{2pt}
\titleformat{\subsection}[block]{\bfseries}{\thesubsection.}{0.5em}{}

\makeatletter
\renewcommand\chapter{%
	\if@openright\cleardoublepage\else\clearpage\fi
	\thispagestyle{plain}%
	\global\@topnum\z@%
	\@afterindentfalse%
	\secdef\@chapter\@schapter}
\makeatother


\makeatletter
\renewcommand{\tableofcontents}{%
	\chapter*{Гарчиг}%
	\addcontentsline{toc}{chapter}{Гарчиг}%
	\@starttoc{toc}%
	\clearpage
}
\makeatother
