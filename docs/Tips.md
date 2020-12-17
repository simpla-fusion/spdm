Tips
==========================


## Build IMAS Data ditionary (only)

    sudo apt-get install xsltproc libsaxonhe-java 
    git clone ssh://git@git.iter.org/imas/data-dictionary.git
    cd data-dictionary
    CLASSPATH=/usr/share/java/Saxon-HE-9.9.1.5.jar SAXONJARFILE=Saxon-HE-9.9.1.5.jar make
    xdg-open html_documentation/html_documentation.html

##　from png to gif

    convert -delay 100 -loop 1 magnetic/*.png magnetic2.gif