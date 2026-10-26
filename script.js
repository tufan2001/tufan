// Mobile navigation
const menuToggle = document.querySelector("#menuToggle");
const navLinks = document.querySelector("#navLinks");

menuToggle.addEventListener("click", () => {
  const isOpen = navLinks.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
  menuToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
  menuToggle.textContent = isOpen ? "×" : "☰";
});

navLinks.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "Open navigation");
    menuToggle.textContent = "☰";
  });
});

// Reveal elements as they enter the viewport
const revealObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

// Rotating role text / typing effect
const roleElement = document.querySelector("#typedRole");
const roles = ["Quality Assurance", "Food Safety", "Production Operations", "Food Processing"];
let roleIndex = 0;
let characterIndex = roles[0].length;
let deleting = true;

function typeRole() {
  const currentRole = roles[roleIndex];

  if (deleting) {
    characterIndex--;
    roleElement.textContent = currentRole.substring(0, characterIndex);
    if (characterIndex <= 0) {
      deleting = false;
      roleIndex = (roleIndex + 1) % roles.length;
      setTimeout(typeRole, 300);
      return;
    }
  } else {
    const nextRole = roles[roleIndex];
    characterIndex++;
    roleElement.textContent = nextRole.substring(0, characterIndex);
    if (characterIndex >= nextRole.length) {
      deleting = true;
      setTimeout(typeRole, 1500);
      return;
    }
  }
  setTimeout(typeRole, deleting ? 45 : 75);
}
setTimeout(typeRole, 1700);

// Automatically keep the copyright year current
document.querySelector("#year").textContent = new Date().getFullYear();
