---
layout: post
title: Print directory structure as tree in Linux
date: '2019-01-23'
cover_image: /content/images/dir-tree.png
---

#### Without files : 

    ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'

#### Output : 

     .
     |-11d9faf945d073033780fd924b2b09ff42155763
     |---assets
     |---variables
     |-77e12a4b00ee2743f22a9dd90b9d66aa8f78faff
     |---assets
     |---variables

#### With files :

    find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"

#### Output : 

     .
     |-77e12a4b00ee2743f22a9dd90b9d66aa8f78faff
     | |-variables
     | | |-variables.data-00000-of-00001
     | | |-variables.index
     | |-assets
     | |-saved_model.pb
     | |-tfhub_module.pb
     |-11d9faf945d073033780fd924b2b09ff42155763
     | |-variables
     | | |-variables.data-00000-of-00001
     | | |-variables.index
     | |-assets
     | |-saved_model.pb
     | |-tfhub_module.pb
