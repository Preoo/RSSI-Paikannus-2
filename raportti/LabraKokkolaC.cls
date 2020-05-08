% Originally written by Antti-Juhani Kaijanaho on December 2002.
% You may treat this file as if it were in the public domain.
% Subsequently modified by Matthieu Weber and Antti-Juhani Kaijanaho.
% Modified by Risto Honkanen Jan, 2013

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                     PLEASE READ THE MANUAL!                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\filedate{2011/10/19}
\def\fileversion{2.6}
\def\fileinfo{JY MIT labrat}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{LabraKokkolaC}[\filedate\space\fileversion\space\fileinfo]
\typeout{LabraKokkolaC <\filedate:\fileversion> - Tietotekniikan laitoksen laboratoriotöille}

%% Some packages need special handling if the output is PDF, this \if is just
%% in case we need it
\newif\iflabra@pdf
\ifx\pdfoutput\undefined
  \labra@pdffalse
\else
  \ifnum\pdfoutput=1
    \labra@pdftrue
  \else
    \labra@pdffalse
  \fi
\fi

\newif\iffi
\fitrue

\newif\ifcopyright
\copyrighttrue

\newif\ifnumbib
\numbibtrue

\newif\ifnottoc
\nottoctrue

\newif\ifoldtitle
\oldtitletrue

\def\labra@baseclass{report}

\newif\ifaltsubsec
\altsubsecfalse

\newif\ifpalatino
\palatinofalse

\newif\iflabra@hyperref
\labra@hyperreffalse

\newif\iflogo
\logofalse

\newif\ifaltlinespread
\altlinespreadfalse

\newif\ifsurnamefirst
\surnamefirstfalse

\def\labra@muistitko#1{\textit{\labra@msg@muistitko} \texttt{\textbackslash #1}\textit{\labra@msg@komentoa}}

\def\labra@title{\labra@muistitko{title}}
\def\labra@author{\labra@muistitko{author}}
\def\labra@date{\today}
%\def\labra@paikka{Kokkolan yliopistokeskus Chydenius}
\def\labra@yliopisto{Jyv\"askyl\"an yliopisto}
\def\labra@university{University of Jyv\"askyl\"a}
\def\labra@laitos{%
  \iffi%
  Informaatioteknologian tiedekunta%
  \else%
  Faculty of Mathematical Information Technology%
  \fi%
}
\def\labra@aine{%
  \iffi%
  Sensoriverkkojen laboratorioty\"ot%
  \else%
  in Information Technology%
  \fi%
}
\def\labra@paikka{%
  \iffi%
  Kokkolan yliopistokeskus Chydenius%
  \else%
  Kokkola University Consortium Chydenius%
  \fi%
}
\def\labra@linja{}

\def\labra@tyyppi{TIES5370}
\def\labra@type{Master's Thesis}
\def\labra@tiivistelma{\labra@muistitko{tiivistelma}}
\def\labra@puhelinnumero{\labra@muistitko{puhelinnumero}}
\def\labra@sivumaara{\labra@muistitko{sivumaara}}
\def\labra@ohjaaja{\labra@muistitko{ohjaaja}}
\def\labra@abstract{\labra@muistitko{abstract}}
\def\labra@titletrans{\labra@muistitko{translatedtitle}}
\def\labra@avainsanat{\labra@muistitko{avainsanat}}
\def\labra@keywords{\labra@muistitko{keywords}}
%\def\labra@yhteystiedot{\labra@muistitko{yhteystiedot}}
\def\labra@copyrightowner{\labra@makeauthorsl}
\def\labra@license{All rights reserved.}
\def\labra@copyrightyear{\number\year}
\def\labra@acmccs{}
\def\labra@ysa{}

\def\labra@authors{}
\let\labra@authorstart\relax
\let\labra@authorsep\relax
\let\labra@authorend\relax

\def\labra@authorsl{}
\let\labra@authorlstart\relax
\let\labra@authorlsep\relax
\let\labra@authorlend\relax

\def\labra@srohtua{}
\let\labra@rohtuastart\relax
\let\labra@rohtuasep\relax
\let\labra@rohtuaend\relax


%% Input encoding
\def\labra@inputenc{latin1}
%\def\labra@inputenc{utf8}
\DeclareOption{ansinew}{\gdef\labra@inputenc{ansinew}}
\DeclareOption{applemac}{\gdef\labra@inputenc{applemac}}
\DeclareOption{ascii}{\gdef\labra@inputenc{ascii}}
\DeclareOption{cp1250}{\gdef\labra@inputenc{cp1250}}
\DeclareOption{cp1252}{\gdef\labra@inputenc{cp1252}}
\DeclareOption{cp437}{\gdef\labra@inputenc{cp437}}
\DeclareOption{cp437de}{\gdef\labra@inputenc{cp437de}}
\DeclareOption{cp850}{\gdef\labra@inputenc{cp850}}
\DeclareOption{cp852}{\gdef\labra@inputenc{cp852}}
\DeclareOption{cp865}{\gdef\labra@inputenc{cp865}}
\DeclareOption{decmulti}{\gdef\labra@inputenc{decmulti}}
\DeclareOption{latin1}{\gdef\labra@inputenc{latin1}}
\DeclareOption{latin2}{\gdef\labra@inputenc{latin2}}
\DeclareOption{latin3}{\gdef\labra@inputenc{latin3}}
\DeclareOption{latin5}{\gdef\labra@inputenc{latin5}}
\DeclareOption{latin9}{\gdef\labra@inputenc{latin9}}
\DeclareOption{next}{\gdef\labra@inputenc{next}}
\DeclareOption{utf8}{\gdef\labra@inputenc{utf8}}

\DeclareOption{copyright}{\copyrighttrue}
\DeclareOption{nocopyright}{\copyrightfalse}

\DeclareOption{numbib}{\numbibtrue}
\DeclareOption{nonumbib}{\numbibfalse}

\DeclareOption{toc}{\nottocfalse}
\DeclareOption{nottoc}{\nottoctrue}

\DeclareOption{oldtitle}{\oldtitletrue}
\DeclareOption{newtitle}{\oldtitlefalse}

%\DeclareOption{shortthesis}{\shortthesistrue}

\DeclareOption{altsubsec}{\altsubsectrue}

\DeclareOption{surnamefirst}{\surnamefirsttrue}

%\DeclareOption{kandi}{%\shortthesistrue%
%  \gdef\labra@tyyppi{kandidaatintutkielma}%
%  \gdef\labra@type{Bachelor's Thesis}%
%}
%
%\DeclareOption{labra}{%\shortthesistrue%
%  \gdef\labra@tyyppi{ laboratorioty\"ot}%
%  \gdef\labra@type{laboratory report}%
%}
%
%\DeclareOption{seminaari}{%\shortthesistrue%
%  \gdef\labra@tyyppi{seminaarikirjoitelma}%
%  \gdef\labra@type{Seminar Essay}%
%}
%
%\DeclareOption{referaatti}{%\shortthesistrue%
%  \gdef\labra@tyyppi{referaatti}%
%  \gdef\labra@type{Extract Content}%
%}

\let\labra@lof\relax
\let\labra@lot\relax
\DeclareOption{lof}{\gdef\labra@lof{\listoffigures}}
\DeclareOption{lot}{\gdef\labra@lot{\listoftables}}


\DeclareOption{finnish}{\fitrue}
\DeclareOption{english}{\fifalse}

\DeclareOption{palatino}{\palatinotrue}

\DeclareOption{logo}{\logotrue}

\DeclareOption{altlinespread}{\altlinespreadtrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\labra@baseclass}}
\ProcessOptions
\LoadClass[a4paper,12pt]{\labra@baseclass}

\iflogo
\iflabra@pdf
\IfFileExists{logo_jyu.pdf}
  {\typeout{Using logo file logo_jyu.pdf}}
  {\logofalse\ClassWarning{labra2}{Could not find logo_jyu.pdf, logo is disabled}}
\else
\IfFileExists{logo_jyu.eps}
  {\typeout{Using logo file logo_jyu.eps}}
  {\logofalse\ClassWarning{labra2}{Could not find logo_jyu.eps, logo is disabled}}
\fi
\fi

%\RequirePackage{color}

  
  \iffi%
  	\bibliographystyle{acmKokkolaFi}%
  \else%
  	\bibliographystyle{acmKokkolaEn}%
  \fi%
  
\RequirePackage[fixlanguage]{babelbib}
\selectbiblanguage{finnish}

\RequirePackage{everyshi}[1994/12/09]
\RequirePackage[\labra@inputenc]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\ifpalatino
\RequirePackage{mathpazo}
\RequirePackage[scaled=.95]{helvet}
\RequirePackage{courier}
\ifaltlinespread
\renewcommand{\baselinestretch}{1.575} % 1.5 * 1.05
\else
\renewcommand{\baselinestretch}{1.26} % 1.2 * 1.05
\fi
\else
\ifaltlinespread
\renewcommand{\baselinestretch}{1.5}
\else
\renewcommand{\baselinestretch}{1.2}
\fi
\fi
\RequirePackage[left=35mm,right=20mm,top=35mm,bottom=35mm,nohead]{geometry}
\RequirePackage[finnish,english]{babel}
\ifnumbib
\ifnottoc
\RequirePackage[numbib,nottoc]{tocbibind}
\else %nottoc
\RequirePackage[numbib]{tocbibind}
\fi %nottoc
\else %numbib
\ifnottoc
\RequirePackage[nottoc]{tocbibind}
\else  %nottoc
\RequirePackage{tocbibind}
\fi %nottoc
\fi %numbib

\iffi
\def\labra@msg@muistitko{Muistitko k\"aytt\"a\"a}
\def\labra@msg@komentoa{-komentoa?}
\def\termlistname{Sanasto}
\def\appendicesname{Liitteet}
\else
\def\labra@msg@muistitko{Did you remember to use the}
\def\labra@msg@komentoa{ command?}
\def\termlistname{Glossary}
\def\appendicesname{Appendices}
\fi

\newcounter{pages}

\iffi
\def\selectdefaultlanguage{\selectlanguage{finnish}}
\else
\def\selectdefaultlanguage{\selectlanguage{english}}
\fi

%% This requires extensive explanation.
%%
%% Defining the command that displays the total number of pages requires to know
%% if hyperref is loaded or not, because it relies on \if-commands that do not
%% exist unless hyperref is loaded.
%%   But hyperref must be loaded by the user, not by labra2, because hyperref
%% overwrites definitions made in many packages. The user may use such
%% packages, therefore hyperref must be loaded *after* these packages. In
%% addition, hyperref must be loaded before packages based on the float package,
%% because the latter is incompatible with hyperref.
%% labra2 therefore expects the user to potentially load hyperref, and if it is
%% the case, it defines \TotPages@putlabel just before \begin{document}. This
%% could not be done with \AtBeginDocument, because hyperref also uses
%% \AtBeginDocument extensively, and labra2' \AtBeginDocument{\def\@schapter...}
%% would have been executed *before* hyperref is loaded (\begin{document} hooks
%% are executed in the same order they are defined, and labra2 would define its
%% hook before hyperref defines its ones).  Defining a hook in a hook i.e.,
%% \AtBeginDocument{\AtBeginDocument{\def\whatever}} does not work, labra2
%% therefore needs to define its own hook, independently from the normal
%% \begin{document} hooks. The proper way of doing so would be to redefine
%% \document, but for some obscure reason, this does not work. The working
%% solution consists in redefining \begin and check if it is \begin{document}
%% (and then execute labra2' own hook before actually executing
%% \begin{document}) or \begin{sometingelse} (then do nothing special).
%%   labra2' own hook is one \AtBeginDocument that puts its own commands after every others
%% (defining \TotPages@putlabel, then calling \maketitle).

%%%% Begin dirty hack
\def\labra@before@begin@document{
\AtBeginDocument{%
  \@ifpackageloaded{hyperref}{
  \labra@hyperreftrue
  \def\toclevel@schapter{0}%
  \def\toclevel@appendix{0}%
  \def\toclevel@algorithm{0}%

   \define@Hy@TotPages@putlabel
   }{%
   \labra@hyperreffalse
   \define@TotPages@putlabel
   }%
   \pagenumbering{roman}
   \selectdefaultlanguage
   \iflogo
   \@ifpackageloaded{graphicx}{}{%
    \ClassWarning{labra2}{The logo option requires the graphicx package to be
    loaded, loading it now.}
    \iflabra@pdf
    \RequirePackage[pdftex]{graphicx}
    \else
    \RequirePackage[dvips]{graphicx}
    \fi
   }
   \fi
   \maketitle
}}

\RequirePackage{ifthen}
\let\labra@begin=\begin
\renewcommand*{\begin}[1]{%
\ifthenelse{\equal{#1}{document}}{%
\labra@before@begin@document
}{}%
\labra@begin{#1}%
}
%%%% End dirty hack

\newcounter{TotPages}
\EveryShipout{\stepcounter{TotPages}}

%% The definition of \TotPages@putlabel must be delayed, since hyperref is not
%% yet loaded at this point, all the \ifHy@xxx are undefined but latex
%% still complains about the corresponding \fi
\def\define@Hy@TotPages@putlabel{
\def\TotPages@putlabel{%
  \addtocounter{page}{-1}%
  \if@filesw
    \begingroup
      \ifHy@pageanchor
        \ifHy@hypertexnames
          \ifHy@plainpages
             \def\Hy@temp{\arabic{page}}%
          \else
             \let\textlatin\@firstofone
             \edef\Hy@temp{\thepage}%
          \fi
        \else
          \def\Hy@temp{\the\Hy@pagecounter}%
        \fi
      \fi
      \immediate\write\@mainaux{%
          \string\newlabel{TotPages}{{\theTotPages}{\thepage}{}{%
          \ifHy@pageanchor page.\Hy@temp\fi
        }{}}%
      }%
    \endgroup
  \fi
  \addtocounter{page}{1}%
}}
\def\define@TotPages@putlabel{
\def\TotPages@putlabel{%
  \addtocounter{page}{-1}%
  \if@filesw
    \begingroup
      \immediate\write\@mainaux{%
          \string\newlabel{TotPages}{{\theTotPages}{\thepage}}%
      }%
    \endgroup
  \fi
  \addtocounter{page}{1}%
}}%

\AtEndDocument{%
\clearpage\TotPages@putlabel
}


\renewcommand{\title}[1]{\gdef\labra@title{#1}}
\renewcommand{\author}[1]{%
 \ClassWarning{labra2}{\noexpand\author is deprecated!}%
 \gdef\labra@authors{#1 \ifoldtitle\else(Use \noexpand\setauthor!)\fi}
 \gdef\labra@srohtua{#1 \ifoldtitle\else(Use \noexpand\setauthor!)\fi}
}
%\newcommand{\setauthor}[2]{\gdef\labra@author{#1 #2}\gdef\labra@rohtua{#2, #1}}
\newcounter{labra@authorcount}
\newcommand{\setauthor}[2]{
\ifsurnamefirst
  \@cons\labra@authors{\labra@authorstart#2\labra@authorsep#1\labra@authorend}
  \@cons\labra@authorsl{\labra@authorlstart#2\labra@authorlsep#1\labra@authorlend}
\else
  \@cons\labra@authors{\labra@authorstart#1\labra@authorsep#2\labra@authorend}
  \@cons\labra@authorsl{\labra@authorlstart#1\labra@authorlsep#2\labra@authorlend}
\fi
  \@cons\labra@srohtua{\labra@rohtuastart#2\labra@rohtuasep#1\labra@rohtuaend}
	\addtocounter{labra@authorcount}{1}%
}

\newcounter{labra@author@count}
\newcounter{labra@authorcount@}
\def\labra@makeauthors{%
  \setcounter{labra@author@count}{\value{labra@authorcount}}
	\setcounter{labra@authorcount@}{0}%
  \gdef\labra@authorstart{
		\addtocounter{labra@author@count}{-1}%
	  \ifthenelse{\value{labra@authorcount@} = 0}{\begin{tabular}{c}}{}
		\addtocounter{labra@authorcount@}{1}%
	}
	\gdef\labra@authorsep{\space}%
	\gdef\labra@authorend{%
	  \ifthenelse{\value{labra@author@count} = 0}{\end{tabular}}{\\[0.2em]}%
	}%
	\labra@authors
}

\newcounter{labra@authorl@count}
\newcounter{labra@authorlcount@}
\def\labra@makeauthorsl{%
  \setcounter{labra@authorl@count}{\value{labra@authorcount}}%
	\setcounter{labra@authorlcount@}{0}%
  \gdef\labra@authorlstart{%
		\addtocounter{labra@authorl@count}{-1}%
	  \ifthenelse{\value{labra@authorlcount@} > 0}{%
		  \ifthenelse{\value{labra@authorl@count} = 0}%
			  {\iffi\space ja\space \else\space and\space \fi}{,\space}%
		}{}%
		\addtocounter{labra@authorlcount@}{1}%
	}%
	\gdef\labra@authorlsep{\space}%
	\gdef\labra@authorlend{\relax}%
	\labra@authorsl
}

\newcounter{labra@rohtua@count}
\def\labra@makesrohtua{
  \setcounter{labra@rohtua@count}{\value{labra@authorcount}}%
  \gdef\labra@rohtuastart{\relax}
	\gdef\labra@rohtuasep{,\space}%
	\gdef\labra@rohtuaend{%
		\addtocounter{labra@rohtua@count}{-1}%
	  \ifthenelse{\value{labra@rohtua@count}=0}{\relax}{;\space}
	}%
	\labra@srohtua
}

\renewcommand{\date}[1]{%
  \ClassWarning{labra2}{\noexpand\date is deprecated!}%
  \gdef\labra@date{#1}%
}
\newcommand{\setdate}[3]{\gdef\day{#1}\gdef\month{#2}\year=#3}
\newcommand{\paikka}[1]{\gdef\labra@paikka{#1}}
\newcommand{\yliopisto}[1]{\gdef\labra@yliopisto{#1}}
\newcommand{\university}[1]{\gdef\labra@university{#1}}
\newcommand{\laitos}[1]{\gdef\labra@laitos{#1}}
\newcommand{\aine}[1]{\gdef\labra@aine{#1}}
\newcommand{\linja}[1]{\gdef\labra@linja{(#1)}}
\newcommand{\tyyppi}[1]{\gdef\labra@tyyppi{#1}}
\newcommand{\type}[1]{\gdef\labra@type{#1}}
\newcommand{\keywords}[1]{\gdef\labra@keywords{\begin{foreignlanguage}{english}#1\end{foreignlanguage}}}
\newcommand{\avainsanat}[1]{\gdef\labra@avainsanat{\begin{foreignlanguage}{finnish}#1\end{foreignlanguage}}}
\newcommand{\contactinformation}[1]{\yhteystiedot{#1}}
%\newcommand{\yhteystiedot}[1]{\gdef\labra@yhteystiedot{#1}}

\newcounter{labra@yhteystiedotcount}
\def\labra@yhteystiedot{}
\let\labra@yhteystiedotstart\relax
\let\labra@yhteystiedotend\relax

\newcommand{\yhteystiedot}[1]{%
  \@cons\labra@yhteystiedot{\labra@yhteystiedotstart#1\labra@yhteystiedotend}
	\addtocounter{labra@yhteystiedotcount}{1}
}
\newcounter{labra@yhteystiedot@count}
\def\labra@makeyhteystiedot{%
  \setcounter{labra@yhteystiedot@count}{\value{labra@yhteystiedotcount}}%
	\gdef\labra@yhteystiedotend{%
		\addtocounter{labra@yhteystiedot@count}{-1}%
	  \ifthenelse{\value{labra@yhteystiedot@count} = 0}{}{,\space}%
	}%
	\labra@yhteystiedot
}

\long\def\tiivistelma#1{\gdef\labra@tiivistelma{\begin{foreignlanguage}{finnish}#1\end{foreignlanguage}}}
\long\def\puhelinnumero#1{\gdef\labra@puhelinnumero{\begin{foreignlanguage}{finnish}#1\end{foreignlanguage}}}
\long\def\sivumaara#1{\gdef\labra@sivumaara{\begin{foreignlanguage}{finnish}#1\end{foreignlanguage}}}
\long\def\ohjaaja#1{\gdef\labra@ohjaaja{\begin{foreignlanguage}{finnish}#1\end{foreignlanguage}}}
\long\def\abstract#1{\gdef\labra@abstract{\begin{foreignlanguage}{english}#1\end{foreignlanguage}}}
\newcommand{\translatedtitle}[1]{\gdef\labra@titletrans{#1}}
\newcommand{\copyrightowner}[1]{\gdef\labra@copyrightowner{#1}}
\newcommand{\license}[1]{\gdef\labra@license{#1}}
\newcommand{\copyrightyear}[1]{\gdef\labra@copyrightyear{#1}}
\newcommand{\acmccs}[1]{\gdef\labra@acmccs{\textbf{ACM Categories and Subject Descriptors:} \begin{foreignlanguage}{english}#1\end{foreignlanguage}\par}}
\newcommand{\ysa}[1]{\gdef\labra@ysa{\textbf{Asiasanat (YSA):} \begin{foreignlanguage}{finnish}#1\end{foreignlanguage}\par}}

\newcommand{\fulltitle}{\labra@title: \labra@tyo}


\newcommand{\labra@toolate}{%
  \ClassError{labra2}{%
    \iffi%
    Bibliografiatietoja ei voi antaa enaa dokumentin alun jalkeen.%
    \else%
    Bibliographical data cannot be given after the beginning of document.%
    \fi%
  }{%
    \iffi%
    Komennot kuten \textbackslash title voi antaa vain ennen
    \textbackslash begin\{document\}-komentoa.
    \else%
    You can give commands like \textbackslash title only before
    the \textbackslash begin\{document\} command.
    \fi%
  }%
}

%\@ifundefined{\csname labra@numpages \endcsname}
%  \def\labra@numpages{\textbf{?}}
%\fi

\newcommand{\mainmatter}{
  \tableofcontents
  \labra@lof
  \labra@lot
%  \setcounter{pages}{\value{page}}
  \setcounter{pages}{0}
  \clearpage
  \pagenumbering{arabic}
  \setcounter{chapter}{0}
}

\renewcommand{\maketitle}{%
\iffi
    \def\labra@tyo{\labra@aine{} %
    \labra@linja{}\if\labra@linja\else\ \fi\labra@tyyppi}%
\else
    \def\labra@tyo{\labra@type{} \labra@aine{} \labra@linja}%
\fi
\ifoldtitle%
  \begin{titlepage}%
    \vspace*{7truecm plus 1truecm minus 3truecm}%
    \centerline{\textbf{\labra@makeauthors}}%
    \vspace{2truecm}%
    \centerline{\Large\textbf{\parbox[top][\height][c]{\textwidth}{\centering \labra@title}}}%
    \vspace{4truecm}%
    \par{%
    \parindent9truecm\parskip0pt%
    \parbox{6.5cm}{\raggedright 
\iffi%
	\hfill
      \labra@aine\ \labra@linja\par%
      \hfill
      \labra@tyyppi%
\else%
	\hfill
      \labra@type\par%
      \hfill
      \labra@aine\ \labra@linja%
\fi%
      \par%
      \hfill
      \labra@date%
      }}%
    \vfill%
\iflogo
    %\center{\hspace{3pt}\includegraphics[height=3.25truecm]{logo_jyu}}%
    % for use with \large and \normalsize instead of \normalsize and \small
    % \labra@yliopisto and \labra@laitos
    \center{\hspace{3pt}\includegraphics[height=2.05truecm]{logo_jyu}}%
\par\vspace{-.6cm}
\begingroup
\ifpalatino\else\fontfamily{ppl}\selectfont\fi
\iffi
    \center{\normalsize\MakeUppercase{\labra@yliopisto}}%
\else
    \center{\normalsize\MakeUppercase{\labra@university}}%
\fi
    \par\vspace{-.5cm}
    \center{\small\MakeUppercase{\labra@laitos}}%
\endgroup
\else
    \vspace{1truecm plus 1truecm minus .5truecm}%
\iffi
    \center{\large\textbf{\labra@yliopisto}}%
\else
    \center{\large\textbf{\labra@university}}%
\fi
    \center{\textbf{\labra@laitos}}%
\fi
    \center{\textbf{\labra@paikka}}%
  \end{titlepage}%
  \newpage%
  \begin{titlepage}%
%\iffi%
%    \noindent\textbf{Tekij\"a\ifthenelse{\value{labra@authorcount}>1}{t}{}:} \labra@makeauthorsl\par%
%    \noindent\textbf{Yhteystiedot:} \labra@makeyhteystiedot\par%
%    \noindent\textbf{Puhelinnumero:} \labra@puhelinnumero\par%
%    \noindent\textbf{Ohjaaja:} \labra@ohjaaja\par%
%    \noindent\textbf{Ty\"on nimi:} {\let\\\relax \labra@title\par}%
%    \noindent\textbf{Title in English:} \labra@titletrans\par%
%    \noindent\textbf{Ty\"o:} \labra@tyo\par%
%   \noindent\textbf{Sivum\"a\"ar\"a:} \labra@sivumaara\par%
%    \noindent\textbf{Tiivistelm\"a:} \labra@tiivistelma\par%
%    \noindent\labra@ysa%
%    \noindent\textbf{Avainsanat:} \labra@avainsanat\par%
%    \noindent\textbf{Abstract:} \labra@abstract\par%
%    \noindent\labra@acmccs%
%    \noindent\textbf{Keywords:} \labra@keywords\par%
%\else%
%    \noindent\textbf{Author\ifthenelse{\value{labra@authorcount}>1}{s}{}:} \labra@makeauthorsl\par%
%    \noindent\textbf{Contact information:} \labra@makeyhteystiedot\par%
%    \noindent\textbf{Phonenumber:} \labra@puhelinnumero\par%
%    \noindent\textbf{Title:} {\let\\\relax \labra@title\par}%
%    \noindent\textbf{Ty\"on nimi:} \labra@titletrans\par%
%    \noindent\textbf{Project:} \labra@tyo\par%
%    \noindent\textbf{Page count:} \labra@sivumaara\par%
%    \noindent\textbf{Abstract:} \labra@abstract\par%
%    \noindent\textbf{Suomenkielinen tiivistelm\"a:} \labra@tiivistelma\par%
%    \noindent\labra@acmccs%
%    \noindent\textbf{Keywords:} \labra@keywords\par%
%    \noindent\labra@ysa%
%    \noindent\textbf{Avainsanat:} \labra@avainsanat\par%
%\fi%
%\ifcopyright%
%    \bigskip%
%    \noindent Copyright \copyright\ \labra@copyrightyear\ \labra@copyrightowner\par%
%    \medskip%
%    \noindent\labra@license%
%\fi%
  \end{titlepage}%
\else%
  % ------------- NEW TITLEPAGE FORMAT -------------
  \begin{titlepage}%
    \vspace*{5truecm plus 2truecm minus 3truecm}%
    \centerline{\textbf{\labra@makeauthorsl}}%
    \vspace{2truecm}%
    \centerline{\Large\textbf{\parbox[top][\height][c]{\textwidth}{\centering \labra@title}}}%
    \vspace{4truecm}%
    \par{%
\iffi%
     \parindent9truecm\parskip0pt%
      \labra@aine\ \labra@linja\par%
      \labra@tyyppi%
\else%
     \parindent6truecm\parskip0pt%
      \labra@type\par%
      \labra@aine\ \labra@linja%
\fi%
      \par%
      \labra@date%
      }%
    \vfill%
    \vspace{1truecm plus 1truecm minus .5truecm}%
\iffi
    \center{\large\textbf{\labra@yliopisto}}%
\else
    \center{\large\textbf{\labra@university}}%
\fi
    \center{\textbf{\labra@laitos}}%
    \center{\textbf{\labra@paikka}}%
  \end{titlepage}%
\ifcopyright%
  \begin{titlepage}%
    \mbox{}\vfill%
    \centering%
    \noindent Copyright \copyright\ \labra@copyrightyear\ \labra@copyrightowner\par%
    \medskip%
    \noindent\labra@license%
    \par\vskip 3 cm plus 1 cm minus 1 cm\par%
    \noindent \labra@yliopisto\\%
    \labra@paikka~\number\year
    %\par\vfill%
  \end{titlepage}%
\fi%
  \iffi%
    \def\labra@finnishtitle{\labra@title}%
    \def\labra@englishtitle{\labra@titletrans}%
  \else%
    \def\labra@finnishtitle{\labra@titletrans}%
    \def\labra@englishtitle{\labra@title}%
  \fi%
  \def\labra@finnishabstract{%
    \begin{titlepage}%
      %\selectlanguage{finnish}%
      \section*{Tiivistelm\"a}%
      \labra@makesrohtua\\%
      {\def\\{ }\labra@finnishtitle{} / \fitrue\labra@makeauthorsl\fifalse}\\%
      \labra@paikka: \labra@yliopisto, \number\year \\%
      \ref{TotPages}~s.\\%
      \labra@tyyppi\par%
      \bigskip%
      \noindent\labra@tiivistelma\par%
      \bigskip%
      \noindent\labra@ysa%
      \noindent\textbf{Avainsanat: } \labra@avainsanat%
      %\selectdefaultlanguage%
    \end{titlepage}%
  }%
  \def\labra@englishabstract{%
    \begin{titlepage}%
      \section*{Abstract}%
      \labra@makesrohtua\\%
      {\def\\{ }\labra@englishtitle{} / \fifalse\labra@makeauthorsl\fitrue}\\%
      \labra@paikka: \labra@university, \number\year \\%
      \ref{TotPages}~p.\\%
      \labra@type\par%
      \bigskip%
      \noindent\labra@abstract\par%
      \bigskip%
      \noindent\labra@acmccs%
      \noindent\textbf{Keywords: } \labra@keywords%
    \end{titlepage}%
  }%
\iffi
  \newpage%
  \labra@finnishabstract%
  \newpage
  \labra@englishabstract%
\else
  \newpage%
  \labra@englishabstract%
  \newpage
  \labra@finnishabstract%
\fi
\fi%
  \global\let\title\labra@toolate%
  \global\let\author\labra@toolate%
  \global\let\date\labra@toolate%  
  \global\let\linja\labra@toolate%  
  \global\let\tyyppi\labra@toolate%  
  \global\let\keywords\labra@toolate%  
  \global\let\linja\labra@toolate%  
  \global\let\tyyppi\labra@toolate%  
  \global\let\keywords\labra@toolate%  
  \global\let\avainsanat\labra@toolate%  
  \global\let\contactinformation\labra@toolate%  
  \global\let\yhteystiedot\labra@toolate%  
  \global\let\tiivistelma\labra@toolate%  
  \global\let\abstract\labra@toolate%  
  \global\let\translatedtitle\labra@toolate%  
  \global\let\copyrightowner\labra@toolate%  
  \global\let\license\labra@toolate%  
  \global\let\copyrightyear\labra@toolate%  
}


\newcommand{\preface}{\chapter*{\prefacename}
\addcontentsline{toc}{chapter}{\prefacename}
}

\newcommand{\termlist}{\chapter*{\termlistname}
\addcontentsline{toc}{chapter}{\termlistname}
}


\renewcommand\appendix{\par

    \addtocontents{toc}{\protect\contentsline{chapter}{\appendicesname}{}{}}
    \setcounter{chapter}{0}%

  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}

\def\@makechapterhead#1{%
  \vspace*{2em}%
  {\parindent \z@ \raggedright \normalfont
    \Large \bfseries 
    \ifnum \c@secnumdepth >\m@ne
      \thechapter\hskip2ex
    \fi
    #1\par\nobreak
    \vskip 1em
  }}
\def\@makeschapterhead#1{%
  \vspace*{2em}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Large \bfseries  #1\par\nobreak
    \vskip 1em
  }}

\renewcommand\section{\@startsection%
  {section}%
  {1}%
  {\z@}%
  {-2em}%
  {1em}%
  {\normalfont\bfseries\large}%
}

\renewcommand\subsection{\@startsection%
  {subsection}%
  {2}%
  {\z@}%
  {-\baselineskip}%
  {\ifaltsubsec
    -2ex
   \else
     2ex
   \fi}% aligns the first line of the following paragraph with the title:
         % bug or feature?
  %{2ex}%
  {\normalfont\bfseries\normalsize}%
}

\renewcommand\subsubsection{\ClassError{labra2}{Don't use sectioning below subsection}{}}
\renewcommand\paragraph{\ClassError{labra2}{Don't use sectioning below subsection}{}}
\renewcommand\subparagraph{\ClassError{labra2}{Don't use sectioning below subsection}{}}

\setcounter{secnumdepth}{2}

\newenvironment{chapterquote}[1]{%
  \begin{quote}\em
  \def\labra@quoteattr{#1}
}{%
  \\\mbox{}\hfill---~\labra@quoteattr
  \end{quote}
  \@endparenv
}

\let\labra@thebibliography\thebibliography
\def\thebibliography#1{%
  \iffi%
    \def\bibname{L\"ahteet}%
  \else%
    \def\bibname{References}%
  \fi%
  \labra@thebibliography{#1}%
}

\RequirePackage{url}

%% Define a new 'leo' style for the package that will use a smaller font.
\makeatletter
\def\url@leostyle{%
  \@ifundefined{selectfont}{\def\UrlFont{\sf}}{\def\UrlFont{\small\ttfamily}}}
\makeatother
%% Now actually use the newly defined style.
\urlstyle{leo}
