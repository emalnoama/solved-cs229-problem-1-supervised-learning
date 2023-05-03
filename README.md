Download Link: https://assignmentchef.com/product/solved-cs229-problem-1-supervised-learning
<br>
<strong>Notes: </strong>(1) These questions require thought, but do not require long answers. Please be as concise as possible. (2) If you have a question about this homework, we encourage you to post your question on our Piazza forum, at https://piazza.com/class#fall2012/cs229. (3) If you missed the first lecture or are unfamiliar with the collaboration or honor code policy, please read the policy on Handout #1 (available from the course website) before starting work. (4) For problems that require programming, please include in your submission a printout of your code (with comments) and any figures that you are asked to plot. (5) Please indicate the submission time and number of late dates clearly in your submission.

<strong>SCPD students: </strong>Please email your solutions to <a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="05667637373c2874644566762b7671646b636a77612b606170">[email protected]</a> with the subject line “Problem Set 1 Submission”. If you are writing your solutions out by hand, please write clearly and in a reasonably large font using a dark pen to improve legibility.

<h1>1.Logistic regression</h1>

(a) [10 points] Consider the log-likelihood function for logistic regression:

<em>m</em>

<em>ℓ</em>(<em>θ</em>) = X<em>y</em><sup>(<em>i</em>) </sup>log<em>h</em>(<em>x</em><sup>(<em>i</em>)</sup>) + (1 − <em>y</em><sup>(<em>i</em>)</sup>)log(1 − <em>h</em>(<em>x</em><sup>(<em>i</em>)</sup>))

<em>i</em>=1

Find the Hessian <em>H </em>of this function, and show that for any vector <em>z</em>, it holds true that

<em>z<sup>T</sup>Hz </em>≤ 0<em>.</em>

[Hint: You might want to start by showing the fact that P<em>i </em>P<em>j z<sub>i</sub>x<sub>i</sub>x<sub>j</sub>z<sub>j </sub></em>= (<em>x<sup>T</sup>z</em>)<sup>2 </sup>≥

0.]

<strong>Remark: </strong>This is one of the standard ways of showing that the matrix <em>H </em>is negative semi-definite, written “<em>H </em>≤ 0.” This implies that <em>ℓ </em>is concave, and has no local maxima other than the global one.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> If you have some other way of showing <em>H </em>≤ 0, you’re also welcome to use your method instead of the one above.

<ul>

 <li>[10 points] On the Leland system, the files /afs/ir/class/cs229/ps/ps1/q1x.dat and /afs/ir/class/cs229/ps/ps1/q1y.dat contain the inputs (<em>x</em><sup>(<em>i</em>) </sup>∈ R<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>) and outputs (<em>y</em><sup>(<em>i</em>) </sup>∈ {0<em>,</em>1}) respectively for a binary classification problem, with one training example per row. Implement<sup>2 </sup>Newton’s method for optimizing <em>ℓ</em>(<em>θ</em>), and apply it to fit a logistic regression model to the data. Initialize Newton’s method with <em>θ </em>= <em>~</em>0 (the vector of all zeros). What are the coefficients <em>θ </em>resulting from your fit? (Remember to include the intercept term.)</li>

 <li>[5 points] Plot the training data (your axes should be <em>x</em><sub>1 </sub>and <em>x</em><sub>2</sub>, corresponding to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or 0). Also plot on the same figure the decision boundary fit by logistic regression. (I.e., this should be a straight line showing the boundary separating the region where <em>h</em>(<em>x</em>) <em>&gt; </em>0<em>.</em>5 from where <em>h</em>(<em>x</em>) ≤ 0<em>.</em>)</li>

</ul>

<h1>2.  Locally weighted linear regression</h1>

Consider a linear regression problem in which we want to “weight” different training examples differently. Specifically, suppose we want to minimize

<em> .</em>

In class, we worked out what happens for the case where all the weights (the <em>w</em><sup>(<em>i</em>)</sup>’s) are the same. In this problem, we will generalize some of those ideas to the weighted setting, and also implement the locally weighted linear regression algorithm.

<ul>

 <li>[2 points] Show that <em>J</em>(<em>θ</em>) can also be written</li>

</ul>

<em>J</em>(<em>θ</em>) = (<em>Xθ </em>− <em>~y</em>)<em><sup>T</sup>W</em>(<em>Xθ </em>− <em>~y</em>)

for an appropriate diagonal matrix <em>W</em>, and where <em>X </em>and <em>~y </em>are as defined in class. State clearly what <em>W </em>is.

<ul>

 <li>[7 points] If all the <em>w</em><sup>(<em>i</em>)</sup>’s equal 1, then we saw in class that the normal equation is</li>

</ul>

<em>X<sup>T</sup>Xθ </em>= <em>X<sup>T</sup>~y,</em>

and that the value of <em>θ </em>that minimizes <em>J</em>(<em>θ</em>) is given by (<em>X<sup>T</sup>X</em>)<sup>−1</sup><em>X<sup>T</sup>~y. </em>By finding the derivative ∇<em>θJ</em>(<em>θ</em>) and setting that to zero, generalize the normal equation to this weighted setting, and give the new value of <em>θ </em>that minimizes <em>J</em>(<em>θ</em>) in closed form as a function of <em>X</em>, <em>W </em>and <em>~y</em>.

<ul>

 <li>[6 points] Suppose we have a training set {(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>); <em>i </em>= 1<em>…,m</em>} of <em>m </em>independent examples, but in which the <em>y</em><sup>(<em>i</em>)</sup>’s were observed with differing variances. Specifically, suppose that</li>

</ul>

I.e., <em>y</em><sup>(<em>i</em>) </sup>has mean <em>θ<sup>T</sup>x</em><sup>(<em>i</em>) </sup>and variance (<em>σ</em><sup>(<em>i</em>)</sup>)<sup>2 </sup>(where the <em>σ</em><sup>(<em>i</em>)</sup>’s are fixed, known, constants). Show that finding the maximum likelihood estimate of <em>θ </em>reduces to solving a weighted linear regression problem. State clearly what the <em>w</em><sup>(<em>i</em>)</sup>’s are in terms of the <em>σ</em>(<em>i</em>)’s.

<ul>

 <li>On the Leland computer system, the files /afs/ir/class/cs229/ps/ps1/q2x.dat and /afs/ir/class/cs229/ps/ps1/q2y.dat contain the inputs (<em>x</em><sup>(<em>i</em>)</sup>) and outputs (<em>y</em><sup>(<em>i</em>)</sup>) for a regression problem, with one training example per row.

  <ol>

   <li>Implement (unweighted) linear regression (<em>y </em>= <em>θ<sup>T</sup>x</em>) on this dataset (using the normal equations), and plot on the same figure the data and the straight line resulting from your fit. (Remember to include the intercept term.)</li>

   <li> Implement locally weighted linear regression on this dataset (using theweighted normal equations you derived in part (b)), and plot on the same figure the data and the curve resulting from your fit. When evaluating <em>h</em>(·) at a query point <em>x</em>, use weights</li>

  </ol></li>

</ul>

<em> ,</em>

with a bandwidth parameter <em>τ </em>= 0<em>.</em>8. (Again, remember to include the intercept term.)

<ul>

 <li>[3 points] Repeat (ii) four times, with <em>τ </em>= 0<em>.</em>1<em>,</em>0<em>.</em>3<em>,</em>2 and 10. Comment <strong>briefly </strong>on what happens to the fit when <em>τ </em>is too small or too large.</li>

</ul>

<ol start="3">

 <li><strong>] Poisson regression and the exponential family </strong>(a) [5 points] Consider the Poisson distribution parameterized by <em>λ</em>:</li>

</ol>

<em>.</em>

Show that the Poisson distribution is in the exponential family, and clearly state what are <em>b</em>(<em>y</em>), <em>η</em>, <em>T</em>(<em>y</em>), and <em>a</em>(<em>η</em>).

<ul>

 <li> Consider performing regression using a GLM model with a Poisson response variable. What is the canonical response function for the family? (You may use the fact that a Poisson random variable with parameter <em>λ </em>has mean <em>λ</em>.)</li>

 <li> For a training set {(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>); <em>i </em>= 1<em>,…,m</em>}, let the log-likelihood of an example be log<em>p</em>(<em>y</em><sup>(<em>i</em>)</sup>|<em>x</em><sup>(<em>i</em>)</sup>;<em>θ</em>). By taking the derivative of the log-likelihood with respect to <em>θ<sub>j</sub></em>, derive the stochastic gradient ascent rule for learning using a GLM model with Poisson responses <em>y </em>and the canonical response function.</li>

 <li><strong> </strong>Consider using GLM with a response variable from any member of the exponential family in which <em>T</em>(<em>y</em>) = <em>y</em>, and the canonical response function for the family. Show that stochastic gradient ascent on the log-likelihood log<em>p</em>(<em>~y</em>|<em>X,θ</em>) results in the update rule <em>θ<sub>i </sub></em>:= <em>θ<sub>i </sub></em>− <em>α</em>(<em>h</em>(<em>x</em>) − <em>y</em>)<em>x<sub>i</sub></em>.</li>

</ul>

<h1>4. [15 points] Gaussian discriminant analysis</h1>

Suppose we are given a dataset {(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>); <em>i </em>= 1<em>,…,m</em>} consisting of <em>m </em>independent examples, where <em>x</em><sup>(<em>i</em>) </sup>∈ R<em><sup>n </sup></em>are <em>n</em>-dimensional vectors, and <em>y</em><sup>(<em>i</em>) </sup>∈ {0<em>,</em>1}. We will model the joint distribution of (<em>x,y</em>) according to:

Here, the parameters of our model are <em>φ</em>, Σ, <em>µ</em><sub>0 </sub>and <em>µ</em><sub>1</sub>. (Note that while there’re two different mean vectors <em>µ</em><sub>0 </sub>and <em>µ</em><sub>1</sub>, there’s only one covariance matrix Σ.)

<ul>

 <li>[5 points] Suppose we have already fit <em>φ</em>, Σ, <em>µ</em><sub>0 </sub>and <em>µ</em><sub>1</sub>, and now want to make a prediction at some new query point <em>x</em>. Show that the posterior distribution of the label at <em>x </em>takes the form of a logistic function, and can be written</li>

</ul>

<em>,</em>

where <em>θ </em>is some appropriate function of <em>φ,</em>Σ<em>,µ</em><sub>0</sub><em>,µ</em><sub>1</sub>. (Note: To get your answer into the form above, for this part of the problem only, you may have to redefine the <em>x</em><sup>(<em>i</em>)</sup>’s to be <em>n </em>+ 1-dimensional vectors by adding the extra coordinate = 1, like we did in class.)

<ul>

 <li>[10 points] For this part of the problem only, you may assume <em>n </em>(the dimension of <em>x</em>) is 1, so that Σ = [<em>σ</em><sup>2</sup>] is just a real number, and likewise the determinant of Σ is given by |Σ| = <em>σ</em><sup>2</sup>. Given the dataset, we claim that the maximum likelihood estimates of the parameters are given by</li>

</ul>

<em>m</em>

<table width="239">

 <tbody>

  <tr>

   <td width="27"><em>φ</em><em>µ</em><sub>0</sub><em>µ</em><sub>1</sub>Σ</td>

   <td width="25">== ==</td>

   <td width="186">1 <em>y</em>(<em>i</em>) = 1<em>m </em>X {                      }<em>i</em>=1<em>m                      </em>{ (<em>i</em>) = 0}<em>x</em>(<em>i</em>)1 <em>y</em><em>i</em>=1<em>m                          </em>{ (<em>i</em>) = 0}1 <em>y</em>P<em>i</em>=1<em>m                      </em>{ (<em>i</em>) = 1}<em>x</em>(<em>i</em>)1 <em>y</em><em>i</em>=11 <em>y</em>P<em>mi</em>=1 { (<em>i</em>) = 1}1 X<em><sup>m    </sup></em>(<em>i</em>) − <em>µ</em><em>y</em>(<em>i</em>))(<em>x</em>(<em>i</em>) − <em>µ</em><em>y</em>(<em>i</em>))<em>T </em>(<em>x</em></td>

  </tr>

 </tbody>

</table>

1

<em>m</em>

<em>i</em>=1

The log-likelihood of the data is

<em>m</em>

<table width="325">

 <tbody>

  <tr>

   <td width="92"><em>ℓ</em>(<em>φ,µ</em><sub>0</sub><em>,µ</em><sub>1</sub><em>,</em>Σ)</td>

   <td width="24">=</td>

   <td width="209">logY<em>p</em>(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>;<em>φ,µ</em><sub>0</sub><em>,µ</em><sub>1</sub><em>,</em>Σ) <em>i</em>=1</td>

  </tr>

  <tr>

   <td width="92"> </td>

   <td width="24">=</td>

   <td width="209"><em>m </em>logY<em>p</em>(<em>x</em><sup>(<em>i</em>)</sup>|<em>y</em><sup>(<em>i</em>)</sup>;<em>µ</em><sub>0</sub><em>,µ</em><sub>1</sub><em>,</em>Σ)<em>p</em>(<em>y</em><sup>(<em>i</em>)</sup>;<em>φ</em>)<em>.</em></td>

  </tr>

 </tbody>

</table>

<em>i</em>=1

By maximizing <em>ℓ </em>with respect to the four parameters, prove that the maximum likelihood estimates of <em>φ</em>, <em>µ</em><sub>0</sub><em>,µ</em><sub>1</sub>, and Σ are indeed as given in the formulas above. (You may assume that there is at least one positive and one negative example, so that the denominators in the definitions of <em>µ</em><sub>0 </sub>and <em>µ</em><sub>1 </sub>above are non-zero.)

(c) <strong> </strong>Without assuming that <em>n </em>= 1, show that the maximum likelihood estimates of <em>φ,µ</em><sub>0</sub><em>,µ</em><sub>1</sub>, and Σ are as given in the formulas in part (b). [Note: If you’re fairly sure that you have the answer to this part right, you don’t have to do part (b), since that’s just a special case.]

<h1>5.  Linear invariance of optimization algorithms</h1>

Consider using an iterative optimization algorithm (such as Newton’s method, or gradient descent) to minimize some continuously differentiable function <em>f</em>(<em>x</em>). Suppose we initialize the algorithm at <em>x</em><sup>(0) </sup>= <em>~</em>0. When the algorithm is run, it will produce a value of <em>x </em>∈ R<em><sup>n </sup></em>for each iteration: <em>x</em><sup>(1)</sup><em>,x</em><sup>(2)</sup><em>,…</em>.

Now, let some non-singular square matrix <em>A </em>∈ R<em><sup>n</sup></em><sup>×<em>n </em></sup>be given, and define a new function <em>g</em>(<em>z</em>) = <em>f</em>(<em>Az</em>). Consider using the same iterative optimization algorithm to optimize <em>g </em>(with initialization <em>z</em><sup>(0) </sup>= <em>~</em>0). If the values <em>z</em><sup>(1)</sup><em>,z</em><sup>(2)</sup><em>,… </em>produced by this method necessarily satisfy <em>z</em><sup>(<em>i</em>) </sup>= <em>A</em><sup>−1</sup><em>x</em><sup>(<em>i</em>) </sup>for all <em>i</em>, we say this optimization algorithm is <strong>invariant to linear reparameterizations</strong>.

<ul>

 <li>[9 points] Show that Newton’s method (applied to find the minimum of a function) is invariant to linear reparameterizations. Note that since <em>z</em><sup>(0) </sup>= <em>~</em>0 = <em>A</em><sup>−1</sup><em>x</em><sup>(0)</sup>, it is sufficient to show that if Newton’s method applied to <em>f</em>(<em>x</em>) updates <em>x</em><sup>(<em>i</em>) </sup>to <em>x</em><sup>(<em>i</em>+1)</sup>, then Newton’s method applied to <em>g</em>(<em>z</em>) will update <em>z</em><sup>(<em>i</em>) </sup>= <em>A</em><sup>−1</sup><em>x</em><sup>(<em>i</em>) </sup>to <em>z</em><sup>(<em>i</em>+1) </sup>= <em>A</em><sup>−1</sup><em>x</em><sup>(<em>i</em>+1)</sup>.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a></li>

 <li>[3 points] Is gradient descent invariant to linear reparameterizations? Justify youranswer.</li>

</ul>