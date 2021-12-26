---
title: "Vision transformers: behind the success, intuition and fallbacks."
date: 2021-11-21T22:59:01+01:00
tags: ["cv", "transformers"]
categories: ["machine learning"]
draft: false
---

![This is an image](https://iaml-it.github.io/posts/2021-04-28-transformers-in-vision/sit.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur at euismod neque, a iaculis eros. Quisque ac luctus metus, convallis tempor turpis. Mauris urna lorem, malesuada eu urna sit amet, sagittis eleifend felis. <!--more--> Phasellus egestas leo sed odio pharetra, vitae lacinia dui pulvinar. Aliquam dictum velit ligula, quis pretium mi luctus a. Vivamus feugiat eros ut ultrices ultrices. Curabitur aliquet, orci semper bibendum interdum, ipsum ante feugiat elit, eu interdum magna ligula non arcu. Duis eget erat sit amet leo commodo ullamcorper non sit amet mi. Duis nec felis vel nisl luctus pharetra. Sed faucibus vestibulum eleifend. Nam sollicitudin elit id felis pharetra, eu commodo dolor fermentum.

{{< highlight go "linenos=table,hl_lines=8 15-17,linenostart=199" >}}
// ... code
{{< / highlight >}}

```python {linenos=table,hl_lines=[8,"15-17"],linenostart=199}
func GetTitleFunc(style string) func(s string) string {
  switch strings.ToLower(style) {
  case "go":
    return strings.Title
  case "chicago":
    return transform.NewTitleConverter(transform.ChicagoStyle)
  default:
    return transform.NewTitleConverter(transform.APStyle)
  }
}
```

