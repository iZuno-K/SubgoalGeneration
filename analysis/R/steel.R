# https://www.trifields.jp/introducing-steel-in-r-1637
# install.packages('mvtnorm')
library(mvtnorm)

steel.test <- function(x, ...) UseMethod("steel.test")

steel.test.default <-
  function(x, g, control = NULL, alternative = c("two.sided", "less", "greater"), ...)
  {
    alternative <- match.arg(alternative)
    if (is.list(x)) {
      if (length(x) < 2L)
        stop("'x' must be a list with at least 2 elements")
      if (!missing(g))
        warning("'x' is a list, so ignoring argument 'g'")
      DNAME <- deparse(substitute(x))
      x <- lapply(x, function(u) u <- u[complete.cases(u)])
      if (!all(sapply(x, is.numeric)))
        warning("some elements of 'x' are not numeric and will be coerced to numeric")
      k <- length(x)
      l <- sapply(x, "length")
      if (any(l == 0L))
        stop("all groups must contain data")
      g <- factor(rep.int(seq_len(k), l))
      x <- unlist(x)
    }
    else {
      if (length(x) != length(g))
        stop("'x' and 'g' must have the same length")
      DNAME <- paste(deparse(substitute(x)), "and",
                     deparse(substitute(g)))
      OK <- complete.cases(x, g)
      x <- x[OK]
      g <- g[OK]
      if (!all(is.finite(g)))
        stop("all group levels must be finite")
      g <- factor(g)
      k <- nlevels(g)
      if (k < 2L)
        stop("all observations are in the same group")
    }
    
    if (is.null(control)) {
      control <- levels(g)[1]
    }
    if (!any(levels(g) == control)) {
      stop("The dataset doesn't contain this control group!")
    }
    
    # calculate ρ
    get.rho <- function(ni)
    {
      l <- length(ni)
      rho <- outer(ni, ni, function(x, y) { sqrt(x/(x+ni[1])*y/(y+ni[1])) })
      diag(rho) <- 0
      return(sum(rho[-1, -1]) / (l - 2) / (l - 1))
    }
    
    ## number of data in each group
    ni <- table(g)
    ## number of group
    a <- length(ni)
    ## data of control group
    xc <- x[g == control]
    ## number of data in control group
    n1 <- length(xc)
    ## decide ρ
    rho <- ifelse(sum(n1 == ni) == a, 0.5, get.rho(ni))
    
    vc <- c()
    vt <- c()
    vp <- c()
    
    for (i in levels(g)) {
      if(control == i) {
        next
      }
      ## ranking group i,j
      r <- rank(c(xc, x[g == i]))
      ## test statistic
      R <- sum(r[1 : n1])
      ## total number of the 2 group data
      N <- n1 + ni[i]
      ## expectation of test statistic
      E <- n1 * (N + 1) / 2
      ## variance of test statistic
      V <- n1 * ni[i] / N / (N - 1) * (sum(r^2) - N * (N + 1)^2 / 4)
      ## t.value
      t <- (R - E) / sqrt(V)
      
      # calculate p.value
      corr <- diag(a - 1)
      corr[lower.tri(corr)] <- rho
      pmvt.lower <- -Inf
      pmvt.upper <- Inf
      if (alternative == "less") {
        pmvt.lower <- -t
        pmvt.upper <- Inf
      }
      else if (alternative == "greater") {
        pmvt.lower <- t
        pmvt.upper <- Inf
      }
      else {
        t <- abs(t)
        pmvt.lower <- -t
        pmvt.upper <- t
      }
      p <- 1 - pmvt(lower = pmvt.lower, upper = pmvt.upper, delta = numeric(a - 1), df = 0, corr = corr, abseps = 0.0001)
      
      vc <- c(vc, paste(i, control, sep = ':'))
      vt <- c(vt, t)
      vp <- c(vp, p)
    }
    df <- data.frame(comparison = vc,
                     t.value = vt,
                     rho = rho,
                     p.value = vp)
    rownames(df) <- NULL
    return(df)
  }

steel.test.formula <-
  function(formula, data, subset, na.action, ...)
  {
    if(missing(formula) || (length(formula) != 3L))
      stop("'formula' missing or incorrect")
    m <- match.call(expand.dots = FALSE)
    if(is.matrix(eval(m$data, parent.frame())))
      m$data <- as.data.frame(data)
    ## need stats:: for non-standard evaluation
    m[[1L]] <- quote(stats::model.frame)
    m$... <- NULL
    mf <- eval(m, parent.frame()) 
    if(length(mf) > 2L)
      stop("'formula' should be of the form response ~ group")
    names(mf) <- NULL
    y <- do.call("steel.test", append(as.list(mf), list(...)))
    y
  }


