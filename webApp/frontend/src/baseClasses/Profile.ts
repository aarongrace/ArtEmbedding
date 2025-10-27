export default class Profile {
  id: string;
  name: string;
  role: string;
  createdDate: Date;
  password: string;
  clan: string | null;
  email: string | null;
  picture: string | null;
  lastLoggedIn: Date | null;

  constructor(
    id: string,
    name: string,
    role: string,
    createdDate: Date,
    password: string,
    clan: string | null = null,
    email: string | null = null,
    picture: string | null = null,
    lastLoggedIn: Date | null = null
  ) {
    this.id = id;
    this.name = name;
    this.role = role;
    this.createdDate = createdDate;
    this.password = password;
    this.clan = clan;
    this.email = email;
    this.picture = picture;
    this.lastLoggedIn = lastLoggedIn;
  }
}