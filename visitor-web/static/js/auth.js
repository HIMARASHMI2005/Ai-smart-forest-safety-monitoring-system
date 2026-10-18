/**
 * Visitor Web Authentication Controller
 */

class VisitorAuth {
  static getToken() {
    return localStorage.getItem("visitor_token");
  }

  static getProfile() {
    const raw = localStorage.getItem("visitor_profile");
    return raw ? JSON.parse(raw) : null;
  }

  static setSession(token, user, profile) {
    localStorage.setItem("visitor_token", token);
    localStorage.setItem("visitor_user", JSON.stringify(user));
    localStorage.setItem("visitor_profile", JSON.stringify(profile));
  }

  static logout() {
    localStorage.removeItem("visitor_token");
    localStorage.removeItem("visitor_user");
    localStorage.removeItem("visitor_profile");
    window.location.href = "/visitor/login";
  }

  static checkAuth() {
    const token = this.getToken();
    if (!token && !window.location.pathname.includes("/login") && !window.location.pathname.includes("/register")) {
      window.location.href = "/visitor/login";
    }
  }
}

window.VisitorAuth = VisitorAuth;
