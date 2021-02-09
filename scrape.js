let users = [];

const webAppScrape = () => {
  const main = document.getElementsByTagName("main");
  if (main.length > 0) {
    main[0].remove();
  }

  // PROBLEM only scrapes the last 80 users. help
  let curr = Array.from(
    document.querySelectorAll(".css-1dbjc4n.r-18u37iz.r-1wbh5a2")
  );
  let filtered = curr
    .map((a) => a.firstChild.innerText)
    .filter(function (val) {
      return /^@/g.test(val);
    });

  for (c in filtered) {
    if (users.indexOf(filtered[c]) === -1) {
      users.push(filtered[c]);
    }
  }

  curr[filtered.length - 1].scrollIntoView();
};

const tweetdeckScrape = () => {
  // tweetdeck scrapes 98 ?
  users = Array.from(
    document.querySelectorAll(
      ".js-column-social-proof.column-detail-level-2.column-panel.flex.flex-column.height-p--100 .account-summary .account-summary-text .account-link.link-complex .username.txt-mute"
    )
  )
    .map((a) => {
      if (a.innerHTML) return a.innerHTML;
    })
    .filter(function (val) {
      return /^@/g.test(val);
    });
};
